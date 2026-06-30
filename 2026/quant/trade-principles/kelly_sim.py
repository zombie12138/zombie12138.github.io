#!/usr/bin/env python3
"""
Kelly criterion: Monte-Carlo growth rate vs the closed-form curve.

For a biased coin (win prob p, net odds b) and bet fraction f, each round
multiplies wealth by (1 + f*b) on a win or (1 - f) on a loss. The long-run
per-round log-growth is

    g(f) = p * ln(1 + f*b) + q * ln(1 - f),    q = 1 - p

maximised at the Kelly fraction f* = (p*b - q) / b.

This script overlays:
  - the theoretical g(f) curve,
  - the Monte-Carlo distribution of realised per-round growth (density heatmap),
  - the simulated median,
all for the SAME (p, b), to show the formula sits on the simulated ridge.

Figure: fig_kelly_growth.png

Requirements: pip install numpy matplotlib
Usage:
  python kelly_sim.py --output-dir ./
  python kelly_sim.py --output-dir ./ -p 0.55 -b 2 --n-rounds 2000
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

DPI = 150
FIGSIZE = (11, 6)


def kelly_fraction(p, b):
    """Optimal bet fraction f* = (p*b - q) / b."""
    return (p * b - (1 - p)) / b


def growth_theory(f, p, b):
    """Closed-form per-round log-growth g(f) (log1p keeps it stable near f=1)."""
    return p * np.log1p(f * b) + (1 - p) * np.log1p(-f)


def _gaussian_smooth(arr, sigma_y, sigma_x):
    """Separable 2D Gaussian blur (pure numpy), to turn the discrete histogram
    comb into a continuous-looking density cloud."""
    def kernel(sigma):
        r = max(int(3 * sigma), 1)
        x = np.arange(-r, r + 1)
        k = np.exp(-0.5 * (x / sigma) ** 2)
        return k / k.sum()

    ky, kx = kernel(sigma_y), kernel(sigma_x)
    out = np.apply_along_axis(lambda c: np.convolve(c, ky, mode='same'), 0, arr)
    out = np.apply_along_axis(lambda r: np.convolve(r, kx, mode='same'), 1, out)
    return out


def simulate_growth(f_grid, p, b, n_paths, n_rounds, rng):
    """Realised per-round log-growth for each f.

    Returns array (len(f_grid), n_paths): each entry is (1/T) * ln(W_T / W_0)
    for one simulated path, which converges to g(f) by the law of large numbers.
    """
    out = np.empty((len(f_grid), n_paths))
    for i, f in enumerate(f_grid):
        wins = rng.random((n_paths, n_rounds)) < p
        step = np.where(wins, np.log1p(f * b), np.log1p(-f))
        out[i] = step.sum(axis=1) / n_rounds
    return out


def fig_growth(outdir, p=0.6, b=1.0, n_paths=8000, n_rounds=150,
               n_f=110, seed=0):
    rng = np.random.default_rng(seed)
    f_star = kelly_fraction(p, b)
    f_max = min(0.95, max(2.5 * f_star, 0.5))
    f_grid = np.linspace(1e-3, f_max, n_f)

    print(f'Simulating: p={p}, b={b}, f*={f_star:.4f}, '
          f'{n_paths} paths x {n_rounds} rounds x {n_f} f-values')
    sim = simulate_growth(f_grid, p, b, n_paths, n_rounds, rng)
    sim_mean = sim.mean(axis=1)        # unbiased estimate of g(f)
    sim_median = np.median(sim, axis=1)
    theory = growth_theory(f_grid, p, b)

    # Density heatmap: y = realised per-round growth, color = density.
    # The per-path growth (k wins out of T) is discrete (T+1 values), so a raw
    # histogram combs into stripes. We trim the loss-streak tails, build a fine
    # histogram, smooth it with a 2D Gaussian to recover a continuous cloud,
    # then normalise each f-column to its own max so the spread shows at every f.
    y_lo = float(np.percentile(sim, 2))
    y_hi = float(np.percentile(sim, 98))
    n_y = 140
    edges = np.linspace(y_lo, y_hi, n_y + 1)
    density = np.zeros((n_y, n_f))
    for i in range(n_f):
        density[:, i], _ = np.histogram(sim[i], bins=edges)

    density = _gaussian_smooth(density, sigma_y=4.0, sigma_x=1.5)
    col_max = density.max(axis=0, keepdims=True)
    density = np.divide(density, col_max, out=np.zeros_like(density),
                        where=col_max > 0)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.imshow(density, origin='lower', aspect='auto',
              extent=[f_grid[0], f_grid[-1], edges[0], edges[-1]],
              cmap='inferno')
    ax.plot(f_grid, theory, color='cyan', lw=2, label='theory $g(f)$')
    ax.plot(f_grid, sim_mean, '--', color='white', lw=1.3,
            label=f'sim mean (T={n_rounds})')
    ax.axhline(0, color='gray', lw=0.6, ls=':')
    ax.axvline(f_star, color='#66ff66', lw=1.4, ls='--',
               label=f'$f^*$ = {f_star:.3f}')
    if 2 * f_star <= f_grid[-1]:
        ax.axvline(2 * f_star, color='#ff6b6b', lw=1.2, ls=':',
                   label=f'$2f^*$ = {2 * f_star:.3f} (g≈0)')

    ax.set_xlabel('bet fraction $f$')
    ax.set_ylabel('per-round log-growth $g$')
    ax.set_title(f'Kelly growth: simulation vs theory (p={p}, b={b})')
    ax.set_xlim(f_grid[0], f_grid[-1])
    ax.set_ylim(edges[0], edges[-1])
    ax.legend(fontsize=9, loc='lower left')
    plt.tight_layout()
    path = os.path.join(outdir, 'fig_kelly_growth.png')
    plt.savefig(path, dpi=DPI)
    plt.close()
    print(f'  saved {path}')
    print(f'  theory peak at f*={f_star:.4f}, g(f*)={growth_theory(f_star, p, b):.5f}')


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output-dir', default='./')
    ap.add_argument('-p', type=float, default=0.6, help='win probability')
    ap.add_argument('-b', type=float, default=1.0, help='net odds (payoff)')
    ap.add_argument('--n-paths', type=int, default=8000)
    ap.add_argument('--n-rounds', type=int, default=150,
                    help='rounds per path; smaller => wider density cloud')
    ap.add_argument('--n-f', type=int, default=110, help='resolution on f axis')
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    fig_growth(args.output_dir, p=args.p, b=args.b, n_paths=args.n_paths,
               n_rounds=args.n_rounds, n_f=args.n_f, seed=args.seed)


if __name__ == '__main__':
    main()
