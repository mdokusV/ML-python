import os
from exceptiongroup import catch
from matplotlib import pyplot as plt
import numpy as np

# TODO: TQDM - pasek postępu

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
# ZAD 1 / ZAD 2
print(
    f"Theta for optimal alpha=0.001, eps=0.001: theta{best_theta}, cost: {history[-1][0]}\n"
)

plt.plot(x, y, "o")
plt.plot(x, hypothesis(best_theta, x))
plt.show()

epsilons = np.linspace(0.000001, 0.001, 50)
cost_x_eps = [[], []]


for ind, eps in enumerate(epsilons):
    if ind % 10 == 0:
        print(ind / len(epsilons) * 100, "%")
    try:
        _, history = gradient_descent(
            hypothesis, cost, [0.0, 0.0], x, y, alpha=0.005, eps=eps
        )
        # print(eps, history[-1][0])
        cost_x_eps[0].append(eps)
        cost_x_eps[1].append(history[-1][0])
    except OverflowError:
        continue
print(100, "%")

# ZAD 3
plt.plot(cost_x_eps[0], cost_x_eps[1])
plt.show()

# ZAD 4
fire = [50, 100, 200]
for i in fire:
    out = hypothesis(best_theta, i)
    print(f"Fire: {i}, theft: {out}")
