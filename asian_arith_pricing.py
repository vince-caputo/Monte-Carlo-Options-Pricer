"""
Arithmetic Asian Call Option — Estimator Comparison

Compares four Monte Carlo estimators for pricing the arithmetic Asian call:
    1. Plain MC
    2. Antithetic variates
    3. Control variates (using geometric Asian call as control)
    4. Quasi-Monte Carlo (Sobol sequences)

Produces three plots:
    asian_prices.png    — Price estimates vs number of paths for all four methods
    asian_errors.png    — Confidence interval widths vs number of paths
    asian_vr.png        — Cumulative average variance reduction factors

Usage:
    python asian_arith_pricing.py [OPTIONS]

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
import numpy as np
import matplotlib.pyplot as plt
import click
from estimators import plain_mc, anti_mc, control_mc, qmc_mc
from payoffs import asian_arith_call, asian_geom_call, asian_geom_closed_form


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

    methods = ["plain", "anti", "control", "qmc"]
    colors = {"plain": "blue", "anti": "red", "control": "green", "qmc": "orange"}
    labels = {
        "plain": "Plain MC",
        "anti": "Antithetic variates",
        "control": "Control variates",
        "qmc": "Quasi-MC (Sobol)",
    }

    prices = {m: np.empty(num_iters) for m in methods}
    errors = {m: np.empty(num_iters) for m in methods}
    vr_factors = {m: np.empty(num_iters) for m in methods}

    control_price = asian_geom_closed_form(S_0, r, sigma, T, K)

    for i in range(num_iters):
        n = path_counts[i]

        price_plain, error_plain = plain_mc(S_0, r, sigma, T, n, num_steps, asian_arith_call, K, rng)
        prices["plain"][i] = price_plain
        errors["plain"][i] = error_plain
        vr_factors["plain"][i] = 1.0

        price_anti, error_anti = anti_mc(S_0, r, sigma, T, n, num_steps, asian_arith_call, K, rng)
        prices["anti"][i] = price_anti
        errors["anti"][i] = error_anti
        vr_factors["anti"][i] = (error_anti / error_plain) ** 2

        price_control, error_control = control_mc(
            S_0, r, sigma, T, n, num_steps,
            asian_arith_call, K,
            asian_geom_call, control_price,
            rng,
        )
        prices["control"][i] = price_control
        errors["control"][i] = error_control
        vr_factors["control"][i] = (error_control / error_plain) ** 2

        price_qmc, error_qmc = qmc_mc(S_0, r, sigma, T, n, num_steps, asian_arith_call, K)
        prices["qmc"][i] = price_qmc
        errors["qmc"][i] = error_qmc
        vr_factors["qmc"][i] = (error_qmc / error_plain) ** 2

    end = time.time()
    print(f"Completed simulations in {end - start:.2f} seconds.")
    print(f"Geometric Asian closed-form (control price): {control_price:.4f}")
    for m in methods:
        print(f"  {labels[m]:30s} price={prices[m][-1]:.4f}  error={errors[m][-1]:.4f}  VR={vr_factors[m][-1]:.3f}")

    # ── Plot 1: Price estimates ──────────────────────────────────────────────
    plt.figure(figsize=(9, 5))
    for m in methods:
        plt.plot(path_counts, prices[m], color=colors[m], label=labels[m])
    plt.xlabel("Number of paths")
    plt.ylabel("Option price")
    plt.title(f"Arithmetic Asian Call — Price Estimates (seed={seed})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/asian_prices.png", dpi=150)
    print("Saved plots/asian_prices.png")

    # ── Plot 2: Confidence interval widths ───────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    max_error = max(np.max(errors[m][1:]) for m in methods)

    for ax, m in zip(axes, methods):
        ax.fill_between(
            path_counts,
            errors[m],
            -errors[m],
            alpha=0.5,
            color=colors[m],
            label=labels[m],
        )
        ax.set_ylim(-max_error, max_error)
        ax.legend(loc="upper right")
        ax.grid(True)

    fig.suptitle(f"Confidence Interval Widths vs. Number of Paths (seed={seed})")
    fig.supxlabel("Number of paths")
    fig.supylabel("CI half-width")
    fig.tight_layout()
    fig.savefig("plots/asian_errors.png", dpi=150)
    print("Saved plots/asian_errors.png")

    # ── Plot 3: Variance reduction factors ──────────────────────────────────
    n_arr = np.arange(1, num_iters + 1)
    vr_methods = ["anti", "control", "qmc"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, m in zip(axes, vr_methods):
        cumavg_vr = np.cumsum(vr_factors[m]) / n_arr
        ax.plot(path_counts, cumavg_vr, color=colors[m])
        ax.axhline(1.0, linestyle="--", color="gray", linewidth=0.8, label="Plain MC baseline")
        ax.set_title(f"{labels[m]}")
        ax.set_xlabel("Number of paths")
        ax.set_ylabel("Variance reduction factor")
        ax.grid(True)
        ax.legend()

    fig.suptitle(f"Cumulative Average Variance Reduction Factors (seed={seed})\n"
                 "(values below 1.0 indicate improvement over plain MC)")
    fig.tight_layout()
    fig.savefig("plots/asian_vr_cumavg.png", dpi=150)
    print("Saved plots/asian_vr_cumavg.png")


if __name__ == "__main__":
    run_demo()
