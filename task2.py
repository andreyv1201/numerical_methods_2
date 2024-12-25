import numpy as np
import matplotlib.pyplot as plt

def solver_task(x0, xn, p, q, f, sigma, gamma, delta, n):
    h = (xn - x0) / n

    def A(i):
        return (1 / (h**2)) - (p(x0 + i * h) / (2 * h))

    def B(i):
        return (1 / (h**2)) + (p(x0 + i * h) / (2 * h))

    def C(i):
        return -(2 / (h**2)) + q(x0 + i * h)

    def F(i):
        return f(x0 + i * h)

    grid = dict()
    alpha = np.empty(n + 1)
    betta = np.empty(n + 1)

    alpha[0] = 0
    betta[0] = 0

    alpha[1] = -gamma[0] / (sigma[0] * h - gamma[0])
    betta[1] = delta[0] / (sigma[0] - gamma[0] / h)

    for i in range(1, n):
        alpha[i + 1] = -B(i) / (A(i) * alpha[i] + C(i))
        betta[i + 1] = (F(i) - A(i) * betta[i]) / (A(i) * alpha[i] + C(i))

    matrix = np.array([[1, -alpha[n]],
                       [1, -(sigma[1] * h + gamma[1]) / gamma[1]]])
    target = np.array([betta[n], -delta[1] * h / gamma[1]])
    ans = np.linalg.solve(matrix, target)

    grid[x0 + (n - 1) * h] = ans[0]
    grid[x0 + n * h] = ans[1]

    for i in range(n - 2, -1, -1):
        grid[x0 + i * h] = alpha[i + 1] * grid[x0 + (i + 1) * h] + betta[i + 1]

    return grid

# Testing program

def p(x):
    return -1 / 2

def q(x):
    return 3

def f(x):
    return 2 * (x**2)

res = solver_task(1, 1.3, p, q, f, [1, 1], [-2, 0], [0.6, 1], 50)
solution = np.array(list(res.items()))
solution = solution[::-1]
plt.scatter(solution[:, 0], solution[:, 1])