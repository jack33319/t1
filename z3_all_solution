#https://brandonrozek.com/blog/obtaining-multiple-solutions-z3/

from z3 import *

# 定義變數
num_cars = 10  # 假設有10輛車
V = [Int(f"V{i}") for i in range(num_cars)]  # 車輛速度
X = [Real(f"X{i}") for i in range(num_cars)]  # 車輛位置

# 建立Z3求解器
solver = Solver()

# Rule 1: V前≥V後
for i in range(num_cars - 1):
    solver.add(V[i] >= V[i+1])

# Rule 2: Minimum speed ≤ Vi ≤ speed limit
minimum_speed = 0
speed_limit = 100
for i in range(num_cars):
    solver.add(And(V[i] >= minimum_speed, V[i] <= speed_limit))

# Rule 3: Xi ≥ Xj → Vi ≥ Vj  ∨  Xi + D ≤ Xj → Vi ≤ Vj
for i in range(num_cars):
    for j in range(num_cars):
        if i != j:
            solver.add(Implies(X[i] >= X[j], V[i] >= V[j]) | Implies(X[i] + D <= X[j], V[i] <= V[j]))

# Rule 4: Xi > Xj → Vi ≥ Vj
for i in range(num_cars):
    for j in range(num_cars):
        if i != j:
            solver.add(Implies(X[i] > X[j], V[i] >= V[j]))

# Rule 5: i = change_line car, j = other car, j ≠ i, 0 ≤ t ≤ 5
# (Vi0 * t + 1/2 * a * t^2 - Vj0 * t - 1/2 * aj * t^2) - (Xi - Xj) ≥ safe_distance
for i in range(num_cars):
    for j in range(num_cars):
        if i != j:
            t = Real(f"t_{i}_{j}")
            Vi0 = Real(f"Vi0_{i}")
            Vj0 = Real(f"Vj0_{j}")
            ai = Real(f"ai_{i}")
            aj = Real(f"aj_{j}")
            safe_distance = RealVal(10)  # 假設安全距離為10
            solver.add((Vi0 * t + 0.5 * ai * t**2 - Vj0 * t - 0.5 * aj * t**2) - (X[i] - X[j]) >= safe_distance)

# Rule 6: Vi - Vi0 ≤ 20
for i in range(num_cars):
    Vi0 = Real(f"Vi0_{i}")
    solver.add(V[i] - Vi0 <= 20)

# 求解並獲取解答
if solver.check() == sat:
    model = solver.model()
    for i in range(num_cars):
        print(f"Car {i}: V{i} = {model[V[i]]}, X{i} = {model[X[i]]}")
else:
    print("No solution found.")
