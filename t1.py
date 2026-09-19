import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import spaces
import carla
import cv2
import time
import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Configuration and data structures
@dataclass
class V2VMessage:
    """Vehicle-to-Vehicle communication message"""
    vehicle_id: int
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    intent: int  # 0: keep lane, 1: change left, 2: change right, 3: slow down
    sensor_data: np.ndarray  # Processed sensor data
    timestamp: float

@dataclass
class Config:
    # CARLA settings
    host: str = 'localhost'
    port: int = 2000
    timeout: float = 10.0
    
    # Environment settings
    num_vehicles: int = 2
    v2v_range: float = 50.0  # meters
    episode_length: int = 1000
    
    # Sensor settings
    camera_width: int = 84
    camera_height: int = 84
    lidar_range: float = 30.0
    
    # RL settings
    action_dim: int = 4  # keep_lane, change_left, change_right, slow_down
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01

# Payoff table for 2-vehicle interactions
class PayoffTable:
    """Game-theoretic payoff table for vehicle interactions"""
    
    def __init__(self):
        # Actions: 0=keep_lane, 1=change_left, 2=change_right, 3=slow_down
        # Payoff matrix: [agent1_action, agent2_action] -> [reward1, reward2]
        self.payoff_matrix = {
            (0, 0): (1.0, 1.0),    # Both keep lane - good
            (0, 1): (0.5, 0.8),    # Agent1 keeps, Agent2 changes left
            (0, 2): (0.5, 0.8),    # Agent1 keeps, Agent2 changes right
            (0, 3): (0.8, 0.3),    # Agent1 keeps, Agent2 slows down
            (1, 0): (0.8, 0.5),    # Agent1 changes left, Agent2 keeps
            (1, 1): (-0.5, -0.5),  # Both change left - conflict
            (1, 2): (0.6, 0.6),    # Agent1 left, Agent2 right - ok
            (1, 3): (0.9, 0.4),    # Agent1 changes left, Agent2 slows
            (2, 0): (0.8, 0.5),    # Agent1 changes right, Agent2 keeps
            (2, 1): (0.6, 0.6),    # Agent1 right, Agent2 left - ok
            (2, 2): (-0.5, -0.5),  # Both change right - conflict
            (2, 3): (0.9, 0.4),    # Agent1 changes right, Agent2 slows
            (3, 0): (0.3, 0.8),    # Agent1 slows, Agent2 keeps
            (3, 1): (0.4, 0.9),    # Agent1 slows, Agent2 changes left
            (3, 2): (0.4, 0.9),    # Agent1 slows, Agent2 changes right
            (3, 3): (0.2, 0.2),    # Both slow down - inefficient
        }
    
    def get_payoff(self, action1: int, action2: int) -> Tuple[float, float]:
        """Get payoff for given action combination"""
        return self.payoff_matrix.get((action1, action2), (0.0, 0.0))

# V2V Communication and Fusion Module
class V2VFusion:
    """Handles V2V message fusion and creates unified representations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.feature_dim = 64  # Dimension of fused features
    
    def create_spatial_graph(self, v2v_messages: List[V2VMessage]) -> np.ndarray:
        """Create a spatial graph representation from V2V messages"""
        if not v2v_messages:
            return np.zeros((self.feature_dim,))
        
        # Simple feature extraction: concatenate normalized positions, velocities, intents
        features = []
        for msg in v2v_messages[:4]:  # Limit to 4 vehicles for consistency
            pos_norm = msg.position[:2] / 100.0  # Normalize position
            vel_norm = np.linalg.norm(msg.velocity[:2]) / 30.0  # Normalize velocity
            intent_one_hot = np.zeros(4)
            intent_one_hot[msg.intent] = 1.0
            
            vehicle_features = np.concatenate([pos_norm, [vel_norm], intent_one_hot])
            features.append(vehicle_features)
        
        # Pad if fewer than 4 vehicles
        while len(features) < 4:
            features.append(np.zeros(7))  # 2 pos + 1 vel + 4 intent
        
        # Flatten and add relative positioning
        fused_features = np.concatenate(features)
        
        # Pad to feature_dim
        if len(fused_features) < self.feature_dim:
            fused_features = np.pad(fused_features, (0, self.feature_dim - len(fused_features)))
        else:
            fused_features = fused_features[:self.feature_dim]
        
        return fused_features
    
    def create_image_representation(self, v2v_messages: List[V2VMessage]) -> np.ndarray:
        """Create an image-based representation from V2V messages"""
        # Create a simple occupancy grid
        grid_size = 32
        image = np.zeros((grid_size, grid_size, 3))  # RGB channels
        
        center = grid_size // 2
        for i, msg in enumerate(v2v_messages[:3]):  # Limit to 3 vehicles
            # Convert world position to grid coordinates
            x_grid = int(center + msg.position[0] / 10.0)  # Scale down
            y_grid = int(center + msg.position[1] / 10.0)
            
            if 0 <= x_grid < grid_size and 0 <= y_grid < grid_size:
                # Different color channels for different vehicles
                image[y_grid, x_grid, i] = 1.0
                # Add intent as intensity
                image[y_grid, x_grid, i] *= (msg.intent + 1) / 4.0
        
        return image

# CARLA Environment Wrapper
class CarlaMultiAgentEnv(gym.Env):
    """Multi-agent CARLA environment with V2V communication"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.payoff_table = PayoffTable()
        self.v2v_fusion = V2VFusion(config)
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(config.action_dim)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.v2v_fusion.feature_dim,), 
            dtype=np.float32
        )
        
        # CARLA connection
        self.client = None
        self.world = None
        self.vehicles = []
        self.sensors = []
        self.step_count = 0
        
        # V2V message storage
        self.v2v_messages = {}
        
    def connect_to_carla(self):
        """Establish connection to CARLA simulator"""
        try:
            self.client = carla.Client(self.config.host, self.config.port)
            self.client.set_timeout(self.config.timeout)
            self.world = self.client.get_world()
            
            # Set synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1
            self.world.apply_settings(settings)
            
            print("Connected to CARLA successfully")
            return True
        except Exception as e:
            print(f"Failed to connect to CARLA: {e}")
            return False
    
    def spawn_vehicles(self):
        """Spawn vehicles in the environment"""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        
        # Get spawn points on highway
        spawn_points = self.world.get_map().get_spawn_points()
        
        for i in range(self.config.num_vehicles):
            if i < len(spawn_points):
                spawn_point = spawn_points[i]
                # Offset vehicles to avoid collision
                spawn_point.location.x += i * 10
                
                vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                self.vehicles.append(vehicle)
                
                # Enable autopilot initially
                vehicle.set_autopilot(True)
        
        print(f"Spawned {len(self.vehicles)} vehicles")
    
    def setup_sensors(self):
        """Setup sensors for each vehicle"""
        blueprint_library = self.world.get_blueprint_library()
        
        for i, vehicle in enumerate(self.vehicles):
            # Camera sensor
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(self.config.camera_width))
            camera_bp.set_attribute('image_size_y', str(self.config.camera_height))
            
            camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
            camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
            
            self.sensors.append(camera)
    
    def get_v2v_messages(self) -> List[V2VMessage]:
        """Collect V2V messages from all vehicles"""
        messages = []
        current_time = time.time()
        
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.is_alive:
                transform = vehicle.get_transform()
                velocity = vehicle.get_velocity()
                
                # Simple intent prediction based on velocity and position
                intent = self.predict_intent(vehicle)
                
                # Dummy sensor data (in real implementation, use actual sensor data)
                sensor_data = np.random.rand(16)  # Placeholder
                
                message = V2VMessage(
                    vehicle_id=i,
                    position=np.array([transform.location.x, transform.location.y, transform.location.z]),
                    velocity=np.array([velocity.x, velocity.y, velocity.z]),
                    intent=intent,
                    sensor_data=sensor_data,
                    timestamp=current_time
                )
                messages.append(message)
        
        return messages
    
    def predict_intent(self, vehicle) -> int:
        """Predict vehicle intent based on current state"""
        # Simple rule-based intent prediction
        velocity = vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2)
        
        if speed < 5.0:
            return 3  # slow_down
        else:
            return 0  # keep_lane (default)
    
    def filter_nearby_messages(self, ego_vehicle, all_messages: List[V2VMessage]) -> List[V2VMessage]:
        """Filter V2V messages based on communication range"""
        ego_pos = ego_vehicle.get_transform().location
        nearby_messages = []
        
        for msg in all_messages:
            distance = np.sqrt(
                (msg.position[0] - ego_pos.x)**2 + 
                (msg.position[1] - ego_pos.y)**2
            )
            if distance <= self.config.v2v_range:
                nearby_messages.append(msg)
        
        return nearby_messages
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Clean up existing actors
        self.cleanup()
        
        # Connect to CARLA if not connected
        if self.client is None:
            if not self.connect_to_carla():
                raise RuntimeError("Cannot connect to CARLA")
        
        # Spawn vehicles and setup sensors
        self.spawn_vehicles()
        self.setup_sensors()
        
        self.step_count = 0
        
        # Get initial observation
        v2v_messages = self.get_v2v_messages()
        if self.vehicles:
            nearby_messages = self.filter_nearby_messages(self.vehicles[0], v2v_messages)
            observation = self.v2v_fusion.create_spatial_graph(nearby_messages)
        else:
            observation = np.zeros(self.v2v_fusion.feature_dim)
        
        info = {"v2v_messages": len(v2v_messages)}
        
        return observation.astype(np.float32), info
    
    def step(self, action):
        """Execute one step in the environment"""
        if not self.vehicles:
            return np.zeros(self.v2v_fusion.feature_dim), 0.0, True, True, {}
        
        # Apply action to the ego vehicle (first vehicle)
        ego_vehicle = self.vehicles[0]
        self.apply_action(ego_vehicle, action)
        
        # Step the world
        self.world.tick()
        self.step_count += 1
        
        # Get V2V messages and create observation
        v2v_messages = self.get_v2v_messages()
        nearby_messages = self.filter_nearby_messages(ego_vehicle, v2v_messages)
        observation = self.v2v_fusion.create_spatial_graph(nearby_messages)
        
        # Calculate reward
        reward = self.calculate_reward(action, v2v_messages)
        
        # Check if episode is done
        terminated = self.step_count >= self.config.episode_length
        truncated = False
        
        # Check for collisions
        if self.check_collision(ego_vehicle):
            reward -= 10.0
            terminated = True
        
        info = {
            "v2v_messages": len(v2v_messages),
            "step_count": self.step_count
        }
        
        return observation.astype(np.float32), reward, terminated, truncated, info
    
    def apply_action(self, vehicle, action):
        """Apply high-level action to vehicle using low-level control"""
        if not vehicle.is_alive:
            return
        
        # Get current vehicle state
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        current_speed = np.sqrt(velocity.x**2 + velocity.y**2)
        
        # Translate high-level action to control commands
        control = carla.VehicleControl()
        
        if action == 0:  # keep_lane
            control.throttle = 0.5
            control.steer = 0.0
        elif action == 1:  # change_left
            control.throttle = 0.4
            control.steer = -0.3
        elif action == 2:  # change_right
            control.throttle = 0.4
            control.steer = 0.3
        elif action == 3:  # slow_down
            control.throttle = 0.0
            control.brake = 0.5
            control.steer = 0.0
        
        vehicle.apply_control(control)
    
    def calculate_reward(self, action, v2v_messages: List[V2VMessage]) -> float:
        """Calculate reward based on action and game-theoretic payoffs"""
        base_reward = 0.1  # Small reward for staying alive
        
        # If we have at least 2 vehicles, use payoff table
        if len(v2v_messages) >= 2:
            ego_action = action
            other_action = v2v_messages[1].intent  # Use intent of nearest vehicle
            
            payoff1, _ = self.payoff_table.get_payoff(ego_action, other_action)
            base_reward += payoff1
        
        # Add speed-based reward
        if v2v_messages:
            ego_speed = np.linalg.norm(v2v_messages[0].velocity[:2])
            speed_reward = min(ego_speed / 20.0, 1.0)  # Reward for maintaining reasonable speed
            base_reward += speed_reward * 0.5
        
        return base_reward
    
    def check_collision(self, vehicle) -> bool:
        """Check if vehicle has collided"""
        # Simple collision detection based on distance to other vehicles
        ego_location = vehicle.get_transform().location
        
        for other_vehicle in self.vehicles[1:]:
            if other_vehicle.is_alive:
                other_location = other_vehicle.get_transform().location
                distance = ego_location.distance(other_location)
                if distance < 3.0:  # Collision threshold
                    return True
        return False
    
    def cleanup(self):
        """Clean up CARLA actors"""
        for sensor in self.sensors:
            if sensor.is_alive:
                sensor.destroy()
        
        for vehicle in self.vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
        
        self.vehicles.clear()
        self.sensors.clear()

# PPO Neural Network
class PPONetwork(nn.Module):
    """PPO actor-critic network for the multi-agent environment"""
    
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action_and_value(self, x, action=None):
        action_logits, value = self.forward(x)
        probs = Categorical(logits=action_logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value

# PPO Training Loop
class PPOTrainer:
    """PPO trainer for the multi-agent CARLA environment"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Environment
        self.env = CarlaMultiAgentEnv(config)
        
        # Network
        obs_dim = self.env.observation_space.shape[0]
        self.network = PPONetwork(obs_dim, config.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.lr)
        
        # Storage
        self.batch_size = 128
        self.num_steps = 256
        self.num_updates = 4
        
        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
    
    def collect_rollouts(self):
        """Collect rollouts from the environment"""
        observations = []
        actions = []
        logprobs = []
        rewards = []
        dones = []
        values = []
        
        obs, _ = self.env.reset()
        
        for step in range(self.num_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, logprob, _, value = self.network.get_action_and_value(obs_tensor)
            
            next_obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            done = terminated or truncated
            
            observations.append(obs)
            actions.append(action.cpu().numpy())
            logprobs.append(logprob.cpu().numpy())
            rewards.append(reward)
            dones.append(done)
            values.append(value.cpu().numpy())
            
            obs = next_obs
            
            if done:
                obs, _ = self.env.reset()
        
        return {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'logprobs': np.array(logprobs),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'values': np.array(values)
        }
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = np.zeros_like(rewards)
        lastgaelam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = 0  # Assuming episode ends
            else:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.config.gamma * self.config.gae_lambda * nextnonterminal * lastgaelam
        
        returns = advantages + values
        return advantages, returns
    
    def update_policy(self, rollouts):
        """Update the policy using PPO"""
        observations = torch.FloatTensor(rollouts['observations']).to(self.device)
        actions = torch.LongTensor(rollouts['actions']).to(self.device)
        old_logprobs = torch.FloatTensor(rollouts['logprobs']).to(self.device)
        
        advantages, returns = self.compute_gae(
            rollouts['rewards'], 
            rollouts['values'], 
            rollouts['dones']
        )
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.num_updates):
            _, new_logprobs, entropy, values = self.network.get_action_and_value(
                observations, actions
            )
            
            # Calculate ratio
            ratio = (new_logprobs - old_logprobs).exp()
            
            # Calculate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values.squeeze(), returns)
            entropy_loss = entropy.mean()
            
            total_loss = policy_loss + self.config.vf_coef * value_loss - self.config.ent_coef * entropy_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item()
        }
    
    def train(self, num_iterations: int = 100):
        """Main training loop"""
        print("Starting PPO training...")
        
        for iteration in range(num_iterations):
            # Collect rollouts
            rollouts = self.collect_rollouts()
            
            # Update policy
            losses = self.update_policy(rollouts)
            
            # Log progress
            avg_reward = np.mean(rollouts['rewards'])
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"  Average Reward: {avg_reward:.3f}")
            print(f"  Policy Loss: {losses['policy_loss']:.3f}")
            print(f"  Value Loss: {losses['value_loss']:.3f}")
            print(f"  Entropy Loss: {losses['entropy_loss']:.3f}")
            print("-" * 50)
            
            # Save model periodically
            if (iteration + 1) % 20 == 0:
                torch.save(self.network.state_dict(), f'ppo_model_iter_{iteration + 1}.pth')
    
    def close(self):
        """Clean up resources"""
        self.env.cleanup()

# Example usage and testing
def main():
    """Main function to run the training"""
    config = Config()
    
    # Initialize trainer
    trainer = PPOTrainer(config)
    
    try:
        # Start training
        trainer.train(num_iterations=50)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")
    finally:
        trainer.close()

if __name__ == "__main__":
    main()
