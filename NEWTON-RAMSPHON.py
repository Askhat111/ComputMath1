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

def newton(f, df, x0, eps=1e-3, Nmax=100):
    if eps <= 0:
        return None, [], "invalid tolerance"
    if Nmax <= 0:
        return None, [], "invalid max iterations"

    table = []
    x = x0

    for n in range(1, Nmax + 1):
        dfx = df(x)
        if dfx == 0:
            return None, table, "division by zero in derivative"

        x_new = x - f(x) / dfx
        fx_new = f(x_new)
        error = abs(x_new - x)
        table.append([n, x_new, fx_new, error])

        if error < eps or abs(fx_new) < eps:
            return x_new, table, "tolerance reached"

        x = x_new

    return x, table, "max iterations reached"

root_n, table_n, reason_n = newton(f, df, x0=1.5, eps=1e-3, Nmax=100)

print("Newton Method")
print("Root estimate:", root_n)
print("Iterations:", len(table_n))
print("Stop reason:", reason_n, "\n")
print_table(table_n)

plot_function_and_root(f, root_n, "Newton Method", xmin=0.0, xmax=2.0)
plot_convergence([r[3] for r in table_n], "Convergence (Newton)")
