import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

def neville_root(x_vals, y_vals, f: Callable = None, plot=True):
    """
    Perform Neville's interpolation to approximate the root (x for which y ≈ 0).

    Args:
        x_vals (list or np.ndarray): x points.
        y_vals (list or np.ndarray): f(x) values.
        f (callable): original function f(x) (optional, for plotting).
        plot (bool): Show plots of interpolation.

    Returns:
        float: Approximated root (x such that f(x) ≈ 0).
    """
    x_vals = np.array(x_vals, dtype=float)
    y_vals = np.array(y_vals, dtype=float)

    n = len(x_vals)
    Q = np.zeros((n, n))
    Q[:, 0] = x_vals  # We're inverting the logic: x(y) instead of y(x)

    # Neville's method solving for x such that y = 0
    for i in range(1, n):
        for j in range(n - i):
            numer = -y_vals[j] * Q[j+1, i-1] + y_vals[j+i] * Q[j, i-1]
            denom = y_vals[j+i] - y_vals[j]
            if abs(denom) < 1e-10:
                raise ZeroDivisionError("Denominator too small during interpolation.")
            Q[j, i] = numer / denom

    root_estimate = Q[0, n-1]

    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))

        # Plot 1: Original function y = f(x)
        if f is not None:
            xs_f = np.linspace(min(x_vals)-1, max(x_vals)+1, 500)
            ys_f = f(xs_f)
            axes[0].plot(xs_f, ys_f, label='f(x)', color='purple')
        axes[0].scatter(x_vals, y_vals, color='red', label='Input points')
        axes[0].axhline(0, color='gray', linestyle='--')
        axes[0].axvline(root_estimate, color='green', linestyle='--', label='Estimated root')
        axes[0].set_title("Original function: y = f(x)")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].legend()
        axes[0].grid()

        # Plot 2: Inverse interpolation x = f⁻¹(y)
        ys = np.linspace(min(y_vals)-1, max(y_vals)+1, 500)
        poly_approx = np.poly1d(np.polyfit(y_vals, x_vals, deg=n-1))
        xs_inv = poly_approx(ys)
        axes[1].plot(ys, xs_inv, label='Interpolated x(y)', color='blue')
        axes[1].scatter(y_vals, x_vals, color='red', label='Input points')
        axes[1].scatter(0, root_estimate, color='green', label='Estimated root x(0)', zorder=5)
        axes[1].axvline(0, color='gray', linestyle='--')
        axes[1].axhline(root_estimate, color='green', linestyle='--')
        axes[1].set_title("Neville's interpolation: x = f⁻¹(y)")
        axes[1].set_xlabel("y")
        axes[1].set_ylabel("x")
        axes[1].legend()
        axes[1].grid()

        plt.tight_layout()
        plt.show()

    print(f"✅ Estimated root (x for which y ≈ 0): {root_estimate:.6f}")
    return root_estimate


# --- Example usage ---

# Fonction test avec racine en sqrt(2)
def f(x): return x**3 + 8*x**2 + 0.5*x - 2

x_vals = [0.2, 0.4, 0.6, 0.8]
y_vals = [f(x) for x in x_vals]

neville_root(x_vals, y_vals, f=f)
