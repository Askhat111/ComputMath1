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

def false_position(f, a, b, eps=1e-3, Nmax=100):
    if a >= b:
        return None, [], "invalid interval order"
    if eps <= 0:
        return None, [], "invalid tolerance"
    if Nmax <= 0:
        return None, [], "invalid max iterations"
    if f(a) * f(b) > 0:
        return None, [], "invalid interval (no sign change)"

    table = []
    fa, fb = f(a), f(b)

    for n in range(1, Nmax + 1):
        if fb - fa == 0:
            return None, table, "division by zero"
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        error = abs(fc)
        table.append([n, c, fc, error])

        if abs(fc) < eps:
            return c, table, "tolerance reached"

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    return c, table, "max iterations reached"


root_f, table_f, reason_f = false_position(f, 1.0, 2.0)

print("False Position Method")
print("Root estimate:", root_f)
print("Iterations:", len(table_f))
print("Stop reason:", reason_f, "\n")
print_table(table_f)

plot_function_and_root(f, root_f, "False Position Method", xmin=0.0, xmax=2.0)
plot_convergence([r[3] for r in table_f], "Convergence (False Position)")
