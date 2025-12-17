import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def f(x):
    return x**3 - x - 1

def df(x):
    return 3*x**2 - 1

def g(x):
    return (x + 1)**(1/3)

def print_table(table, header=("n", "x_n", "f(x_n)", "error")):
    print(f"{header[0]}\t{header[1]}\t\t{header[2]}\t\t{header[3]}")
    for r in table:
        print(f"{r[0]}\t{r[1]:.6f}\t{r[2]:+.6f}\t{r[3]:.6f}")

def plot_function_and_root(f, root, title, xmin=0, xmax=2.0):
    X = np.linspace(xmin, xmax, 400)
    plt.figure(figsize=(7,5))
    plt.plot(X, f(X))
    plt.axhline(0, color="black")
    if root is not None:
        plt.scatter(root, f(root), color="red")
        plt.text(root, f(root)+0.02, f"root ≈ {root:.6f}", ha="center")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.show()

def plot_convergence(errors, title, ylabel="Error"):
    plt.figure(figsize=(6,4))
    plt.plot(errors, marker='o')
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.show()

def bisection(f, a, b, eps=1e-3, Nmax=100):
    if a >= b:
        return None, [], "invalid interval order"
    if eps <= 0:
        return None, [], "invalid tolerance"
    if Nmax <= 0:
        return None, [], "invalid max iterations"
    if f(a) * f(b) > 0:
        return None, [], "invalid interval (no sign change)"

    table = []

    for n in range(1, Nmax + 1):
        c = (a + b) / 2
        fc = f(c)
        error = (b - a) / 2
        table.append([n, c, fc, error])

        if error < eps or abs(fc) < eps:
            return c, table, "tolerance reached"

        if f(a) * fc < 0:
            b = c
        else:
            a = c

    return c, table, "max iterations reached"

root, table_b, reason = bisection(f, 1.0, 2.0)

print("Bisection Method")
print("Root estimate:", root)
print("Iterations:", len(table_b))
print("Stop reason:", reason, "\n")

print_table(table_b)

plot_function_and_root(f, root, "Bisection Method", xmin=0.0, xmax=2.0)
plot_convergence([r[3] for r in table_b], "Convergence (Bisection)")
