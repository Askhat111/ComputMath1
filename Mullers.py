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

def muller(f, x0, x1, x2, eps=1e-3, Nmax=100):

    # basic input checks
    if not (x0 < x1 < x2):
        return None, [], "invalid initial points order"
    if eps <= 0:
        return None, [], "invalid tolerance"
    if Nmax <= 0:
        return None, [], "invalid max iterations"

    table = []

    for n in range(1, Nmax + 1):
        f0, f1, f2 = f(x0), f(x1), f(x2)
        h0, h1 = x1 - x0, x2 - x1

        if h0 == 0 or h1 == 0:
            return None, table, "division by zero in h"

        d0 = (f1 - f0) / h0
        d1 = (f2 - f1) / h1

        a = (d1 - d0) / (h1 + h0)
        b = a * h1 + d1
        c = f2

        D2 = b*b - 4*a*c
        if D2 < 0:
            return None, table, "negative discriminant"

        D = np.sqrt(D2)
        denom = b + D if abs(b + D) > abs(b - D) else b - D
        if denom == 0:
            return None, table, "division by zero in step"

        x3 = x2 - 2*c / denom
        error = abs(x3 - x2)
        table.append([n, x3, f(x3), error])

        if error < eps:
            return x3, table, "tolerance reached"

        x0, x1, x2 = x1, x2, x3

    return x3, table, "max iterations reached"

root_m, table_m, reason_m = muller(f, 1.0, 1.3, 1.5)

print("Muller Method")
print("Root estimate:", root_m)
print("Iterations:", len(table_m))
print("Stop reason:", reason_m, "\n")
print_table(table_m)

plot_function_and_root(f, root_m, "Muller Method", xmin=0.0, xmax=2.0)
plot_convergence([r[3] for r in table_m], "Convergence (Muller)")

