import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize

# Определение задачи многокритериальной оптимизации
problem = get_problem("dtlz2")

algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               verbose=True)

print("Best solutions found:")
print(res.F)

# Построение фронта Парето
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(res.F[:, 0], res.F[:, 1], res.F[:, 2])

ax.set_xlabel('Objective 1')
ax.set_ylabel('Objective 2')
ax.set_zlabel('Objective 3')
ax.set_title('Фронт Парето для задачи DTLZ2 при использовании NSGA-II')
plt.savefig('pareto_front.png')
plt.show()
