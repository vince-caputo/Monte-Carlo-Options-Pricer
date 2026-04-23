"""
Convergence Rate Analysis

Empirically estimates the convergence rate of each Monte Carlo estimator
by fitting a power law of the form:

    error(n) = C / n^alpha

to the observed error data. On a log-log plot this becomes a straight line
with slope -alpha. Theoretical values:
    Plain MC:  alpha = 0.5  (the 1/sqrt(n) rate from the CLT)
    QMC:       alpha ~ 1.0  (the approximate 1/n rate from low-discrepancy theory)
    Antithetic / Control variates: alpha = 0.5 (same rate, lower constant C)

The key insight is that variance reduction techniques (antithetic, control)
improve the constant C but not the convergence rate alpha, while QMC improves
alpha itself. This distinction is visible in the log-log plots as:
    - parallel lines with different intercepts (variance reduction)
    - a steeper slope (QMC)

Analysis is run on both European calls and arithmetic Asian calls.

Usage:
    python convergence_analysis.py [OPTIONS]

Options:
    --num-steps     Number of time steps per GBM path (default: 52)
    --num-paths     Maximum number of paths (default: 50000)
    --num-iters     Number of experiments (default: 50)
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
from scipy.optimize import curve_fit
from estimators import plain_mc, anti_mc, control_mc, qmc_mc
from payoffs import (
    euro_call, euro_call_closed_form,
    asian_arith_call, asian_geom_call, asian_geom_closed_form,
)


def power_law(n, C, alpha):
    """Power law model: error = C / n^alpha."""
    return C / np.power(n, alpha)


def fit_convergence_rate(path_counts, errors):
    """
    Fit a power law to observed (path_count, error) data.

    Returns
    -------
    C : float
        Scale constant.
    alpha : float
        Estimated convergence rate exponent.
    """
    # Use only path counts > 0 and errors > 0 to avoid log(0)
    mask = (path_counts > 0) & (errors > 0)
    popt, _ = curve_fit(
        power_law,
        path_counts[mask],
        errors[mask],
        p0=[1.0, 0.5],
        maxfev=5000,
    )
    return popt[0], popt[1]


def run_analysis(
    option_label, payoff_func, true_price_func, true_price_args,
    control_payoff_func, control_price,
    path_counts, num_steps, S_0, r, sigma, T, K, seed, rng,
    output_prefix,
):
    """
    Run convergence analysis for one option type and produce plots.
    """
    num_iters = len(path_counts)
    methods = ["plain", "anti", "control", "qmc"]
    colors = {"plain": "blue", "anti": "red", "control": "green", "qmc": "orange"}
    labels = {
        "plain": "Plain MC",
        "anti": "Antithetic variates",
        "control": "Control variates",
        "qmc": "Quasi-MC (Sobol)",
    }

    errors = {m: np.empty(num_iters) for m in methods}

    print(f"\n{'='*60}")
    print(f"  {option_label}")
    print(f"{'='*60}")

    for i, n in enumerate(path_counts):
        _, e_plain = plain_mc(S_0, r, sigma, T, n, num_steps, payoff_func, K, rng)
        _, e_anti = anti_mc(S_0, r, sigma, T, n, num_steps, payoff_func, K, rng)
        _, e_control = control_mc(
            S_0, r, sigma, T, n, num_steps,
            payoff_func, K,
            control_payoff_func, control_price,
            rng,
        )
        _, e_qmc = qmc_mc(S_0, r, sigma, T, n, num_steps, payoff_func, K)
        errors["plain"][i] = e_plain
        errors["anti"][i] = e_anti
        errors["control"][i] = e_control
        errors["qmc"][i] = e_qmc

    # Fit convergence rates
    rates = {}
    for m in methods:
        C, alpha = fit_convergence_rate(path_counts, errors[m])
        rates[m] = (C, alpha)
        print(f"  {labels[m]:30s}  alpha = {alpha:.3f}  (theoretical: {'0.5' if m != 'qmc' else '~1.0'})")

    # ── Log-log convergence plot ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))

    for m in methods:
        ax.loglog(path_counts, errors[m], "o", color=colors[m], markersize=3, alpha=0.5)
        C, alpha = rates[m]
        fitted = power_law(path_counts, C, alpha)
        ax.loglog(
            path_counts, fitted,
            color=colors[m],
            label=f"{labels[m]}  (α = {alpha:.3f})",
            linewidth=2,
        )

    # Reference lines for theoretical rates
    n_ref = path_counts.astype(float)
    scale_half = errors["plain"][0] * (path_counts[0] ** 0.5)
    scale_one = errors["qmc"][0] * path_counts[0]
    ax.loglog(n_ref, scale_half / np.sqrt(n_ref), "k--", linewidth=1, alpha=0.4, label="Theoretical O(1/√n)")
    ax.loglog(n_ref, scale_one / n_ref, "k:", linewidth=1, alpha=0.4, label="Theoretical O(1/n)")

    ax.set_xlabel("Number of paths (log scale)")
    ax.set_ylabel("95% CI half-width (log scale)")
    ax.set_title(f"{option_label} — Convergence Rate Analysis (seed={seed})\n"
                 "Slope of fitted line = estimated convergence rate α")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"plots/{output_prefix}_convergence.png", dpi=150)
    print(f"  Saved plots/{output_prefix}_convergence.png")

    # ── Variance reduction vs plain MC ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    vr_methods = ["anti", "control", "qmc"]
    for m in vr_methods:
        vr = (errors[m] / errors["plain"]) ** 2
        ax.plot(path_counts, vr, color=colors[m], label=labels[m])
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=0.8, label="Plain MC baseline")
    ax.set_xlabel("Number of paths")
    ax.set_ylabel("Variance reduction factor\n(lower = better)")
    ax.set_title(f"{option_label} — Variance Reduction vs Plain MC (seed={seed})")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"plots/{output_prefix}_convergence_vr.png", dpi=150)
    print(f"  Saved plots/{output_prefix}_convergence_vr.png")

    return rates


@click.command()
@click.option("--num-steps", "num_steps", default=52, help="Number of steps per GBM path")
@click.option("--num-paths", "num_paths", default=50_000, help="Max number of paths")
@click.option("--num-iters", "num_iters", default=50, help="Number of experiments")
@click.option("--S_0", "S_0", default=100.0, help="Initial stock price")
@click.option("--T", "T", default=1.0, help="Time to maturity in years")
@click.option("--r", "r", default=0.08, help="Risk-free rate")
@click.option("--sigma", "sigma", default=0.3, help="Volatility")
@click.option("--K", "K", default=100.0, help="Strike price")
@click.option("--seed", "seed", default=None, help="Optional RNG seed")
def run_demo(num_steps, num_paths, num_iters, S_0, T, r, sigma, K, seed):
    start = time.time()

    rng = np.random.default_rng(int(seed)) if seed is not None else np.random.default_rng()

    path_counts = np.unique(
        np.geomspace(num_paths // num_iters, num_paths, num=num_iters, dtype=int)
    )

    # Precompute control prices
    euro_true = euro_call_closed_form(S_0, r, sigma, T, K)
    geom_asian_price = asian_geom_closed_form(S_0, r, sigma, T, K)

    print(f"Black-Scholes European call price:     {euro_true:.4f}")
    print(f"Geometric Asian closed-form price:     {geom_asian_price:.4f}")

    # European call analysis
    # For European call, use euro_call itself as control with its closed-form.
    # In practice plain MC, anti, control all use the same payoff;
    # for the control variate on European calls we use the same payoff
    # with the known closed-form price (a degenerate but illustrative case).
    # More naturally, we just run all four estimators and compare rates.
    euro_rates = run_analysis(
        option_label="European Call Option",
        payoff_func=euro_call,
        true_price_func=euro_call_closed_form,
        true_price_args=(S_0, r, sigma, T, K),
        control_payoff_func=euro_call,   # same payoff; control_price below acts as known anchor
        control_price=euro_true,
        path_counts=path_counts,
        num_steps=num_steps,
        S_0=S_0, r=r, sigma=sigma, T=T, K=K,
        seed=seed, rng=rng,
        output_prefix="euro",
    )

    # Arithmetic Asian call analysis
    asian_rates = run_analysis(
        option_label="Arithmetic Asian Call Option",
        payoff_func=asian_arith_call,
        true_price_func=None,
        true_price_args=None,
        control_payoff_func=asian_geom_call,
        control_price=geom_asian_price,
        path_counts=path_counts,
        num_steps=num_steps,
        S_0=S_0, r=r, sigma=sigma, T=T, K=K,
        seed=seed, rng=rng,
        output_prefix="asian",
    )

    end = time.time()
    print(f"\nCompleted convergence analysis in {end - start:.2f} seconds.")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\nSummary of estimated convergence rates (alpha):")
    print(f"  {'Method':<30} {'European Call':>15} {'Asian Arith Call':>18}")
    print("  " + "-" * 65)
    method_labels = {
        "plain": "Plain MC",
        "anti": "Antithetic variates",
        "control": "Control variates",
        "qmc": "Quasi-MC (Sobol)",
    }
    for m in ["plain", "anti", "control", "qmc"]:
        e_alpha = euro_rates[m][1]
        a_alpha = asian_rates[m][1]
        print(f"  {method_labels[m]:<30} {e_alpha:>15.3f} {a_alpha:>18.3f}")
    print(f"\n  Theoretical plain/anti/control: alpha = 0.5")
    print(f"  Theoretical QMC (low dim):      alpha ~ 1.0")
    print(f"  Note: QMC advantage diminishes as num_steps increases")
    print(f"  (current num_steps={num_steps}; for num_steps >> 30, QMC gains are reduced)")


if __name__ == "__main__":
    run_demo()
