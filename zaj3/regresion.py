import os
from exceptiongroup import catch
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd

DIR = os.path.dirname(__file__)


def join_path(filename: str) -> str:
    return os.path.join(DIR, filename)


data = pd.read_csv(join_path("fires_thefts.csv"), header=None, names=["x", "y"])

x = data["x"].to_numpy()
y = data["y"].to_numpy()


def hypothesis(theta, x):
    return theta[0] + theta[1] * x


def cost(h, theta, x, y):
    m = len(y)
    return 1.0 / (2 * m) * sum((h(theta, x[i]) - y[i]) ** 2 for i in range(m))


def gradient_descent(h, cost_fun, theta, x, y, alpha, eps):
    current_cost = cost_fun(h, theta, x, y)
    history = [[current_cost, theta]]
    m = len(y)
    while True:
        try:
            new_theta = [
                theta[0] - alpha / m * sum(h(theta, x[i]) - y[i] for i in range(m)),
                theta[1]
                - alpha / m * sum((h(theta, x[i]) - y[i]) * x[i] for i in range(m)),
            ]
            theta = new_theta

            prev_cost = current_cost
            current_cost = cost_fun(h, theta, x, y)
            if abs(prev_cost - current_cost) <= eps:
                break
            history.append([current_cost, theta])
        except Exception:
            raise OverflowError

    return theta, history


best_theta, history = gradient_descent(
    hypothesis, cost, [0.0, 0.0], x, y, alpha=0.001, eps=0.001
)
print(len(history), history[-1])

# plt.plot(x, y, "o")
# plt.plot(x, hypothesis(best_theta, x))
# plt.show()

epsilons = np.linspace(0.0001, 0.1, 100)
cost_x_eps = []

for eps in epsilons:
    try:
        _, history = gradient_descent(
            hypothesis, cost, [0.0, 0.0], x, y, alpha=0.001, eps=eps
        )
        # print(eps, history[-1][0])
        cost_x_eps.append([eps, history[-1][0]])
    except OverflowError:
        break

plt.plot(cost_x_eps[:][0], cost_x_eps[:][1])
plt.show()
