"""
European Call Option — Plain Monte Carlo Validation

Demonstrates convergence of the plain Monte Carlo estimator to the
Black-Scholes closed-form price as the number of simulated paths increases.

Usage:
    python euro_call_plain_mc.py [OPTIONS]

Options:
    --num-steps     Number of time steps per GBM path (default: 52)
    --num-paths     Maximum number of paths to simulate (default: 10000)
    --num-iters     Number of experiments (default: 100)
    --S_0           Initial stock price (default: 100.0)
    --T             Time to maturity in years (default: 1.0)
    --r             Risk-free rate (default: 0.08)
    --sigma         Volatility (default: 0.3)
    --K             Strike price (default: 100.0)
    --seed          Optional RNG seed for reproducibility
"""

import time
import click
import numpy as np
import matplotlib.pyplot as plt
from estimators import plain_mc
from payoffs import euro_call, euro_call_closed_form


@click.command()
@click.option("--num-steps", "num_steps", default=52, help="Number of steps per GBM path")
@click.option("--num-paths", "num_paths", default=10**4, help="Max number of paths")
@click.option("--num-iters", "num_iters", default=100, help="Number of experiments")
@click.option("--S_0", "S_0", default=100.0, help="Initial stock price")
@click.option("--T", "T", default=1.0, help="Time to maturity in years")
@click.option("--r", "r", default=0.08, help="Risk-free rate")
@click.option("--sigma", "sigma", default=0.3, help="Volatility")
@click.option("--K", "K", default=100.0, help="Strike price")
@click.option("--seed", "seed", default=None, help="Optional RNG seed")
def run_demo(num_steps, num_paths, num_iters, S_0, T, r, sigma, K, seed):
    start = time.time()

    path_counts = np.arange(
        start=num_paths // num_iters,
        stop=num_paths + 1,
        step=num_paths // num_iters,
    )

    rng = np.random.default_rng(int(seed)) if seed is not None else np.random.default_rng()

    prices = np.empty(shape=num_iters)
    errors = np.empty(shape=num_iters)

    for i in range(num_iters):
        price, error = plain_mc(S_0, r, sigma, T, path_counts[i], num_steps, euro_call, K, rng)
        prices[i] = price
        errors[i] = error

    closed_form_price = euro_call_closed_form(S_0, r, sigma, T, K)
    end = time.time()
    print(f"Completed simulation in {end - start:.2f} seconds.")
    print(f"Black-Scholes price:      {closed_form_price:.4f}")
    print(f"Monte Carlo price ({num_paths} paths): {prices[-1]:.4f} ± {errors[-1]:.4f}")

    plt.figure(figsize=(9, 5))
    plt.plot(path_counts, prices, color=(0.584, 0.286, 0.949), label="Plain MC estimate")
    plt.fill_between(
        path_counts,
        prices - errors,
        prices + errors,
        alpha=0.25,
        color=(0.584, 0.286, 0.949),
        label="95% confidence interval",
    )
    plt.axhline(closed_form_price, linestyle="--", color=(0.09, 0.039, 0.529), label="Black-Scholes closed form")
    plt.title(f"European Call — Plain MC Convergence (seed={seed})")
    plt.xlabel("Number of paths")
    plt.ylabel("Option price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/euro_price.png", dpi=150)
    print("Saved plots/euro_price.png")


if __name__ == "__main__":
    run_demo()
