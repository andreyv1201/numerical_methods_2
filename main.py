import numpy as np
import matplotlib.pyplot as plt

def runge_kutta2(x0, y0, f, l, n):
    h = l / n
    grid = dict()
    grid[x0] = y0
    for i in range(n):
        x = x0 + h * i
        y = grid[x]
        fxy = f(x, y)
        grid[x0 + h * (i + 1)] = y + (h / 2) * (fxy + f(x + h, y + h * fxy))
    return grid


def runge_kutta4(x0, y0, f, l, n):
    h = l / n
    grid = dict()
    grid[x0] = y0
    for i in range(n):
        x = x0 + h * i
        y = grid[x]
        k1 = f(x, y)
        k2 = f(x + h / 2, y + (h / 2) * k1)
        k3 = f(x + h / 2, y + (h / 2) * k2)
        k4 = f(x + h, y + h * k3)
        grid[x0 + h * (i + 1)] = y + (h / 6) * (k1 + 2 * (k2 + k3) + k4)
    return grid


def runge_kutta_system4(x0, y0, f, l, n):
    num = y0.shape[0]
    h = l / n
    grid = dict()
    grid[x0] = y0
    for i in range(n):
        x = x0 + h * i
        y = grid[x]
        k1 = np.empty(num)
        k2 = np.empty(num)
        k3 = np.empty(num)
        k4 = np.empty(num)
        for j in range(num):
            k1[j] = f[j](x, *y)
        for j in range(num):
            k2[j] = f[j](x + h / 2, *(y + (h / 2) * k1))
        for j in range(num):
            k3[j] = f[j](x + h / 2, *(y + (h / 2) * k2))
        for j in range(num):
            k4[j] = f[j](x + h, *(y + h * k3))
        grid[x0 + h * (i + 1)] = y + (h / 6) * (k1 + 2 * (k2 + k3) + k4)
    return grid


# Testing part

def f(x, y):
    return -y - x**2


def ans(x):
    return -x**2 + 2 * x - 2 + 12 * np.exp(-x)


grid = runge_kutta2(0, 10, f, 5, 25)
x = list(grid.keys())
y = list(grid.values())
plt.scatter(x, y, s=60, color='blue')

ytrue1 = []
for i in x:
    ytrue1.append(ans(i))
z1 = np.sqrt(np.sum((np.array(ytrue1) - np.array(y))**2))

grid = runge_kutta2(0, 10, f, 5, 60)
x2 = list(grid.keys())
y2 = list(grid.values())
plt.scatter(x2, y2, s=25, color='red')

ytrue2 = []
for i in x2:
    ytrue2.append(ans(i))
plt.plot(x2, ytrue2, color='green', linewidth=2)

z2 = np.sqrt(np.sum((np.array(ytrue2) - np.array(y2))**2))
print(z1, z2)
