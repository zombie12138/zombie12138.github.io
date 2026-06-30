#!/usr/bin/env python3
"""
Gambler's-ruin escape: animated population chart over win probability p.

Fixed-stake betting against an infinite house. For each p we run a population
of M players (start bankroll N0, bet 1 each round) and watch, as rounds roll
forward, what fraction is busted / alive-losing / alive-winning. A cliff forms
at p = 1/2: to its left everyone busts (P=1), to its right the busted fraction
settles at the theoretical (q/p)^{N0}.

Output:
  fig_ruin_population.gif  - animation, rounds scrolling to near-convergence

Requirements: pip install numpy matplotlib pillow
Usage: python ruin_sim.py --output-dir ./
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

DPI = 120
C_BUST = '#e74c3c'      # busted
C_LOSS = '#f39c12'      # alive, below start
C_PROFIT = '#2ecc71'    # alive, above start


def simulate(p_grid, n_players, n0, n_rounds, frame_rounds, seed=0):
    """Population simulation, vectorised over (p, player).

    Returns dict round -> (busted, loss, profit) fraction arrays over p_grid.
    """
    rng = np.random.default_rng(seed)
    P, M = len(p_grid), n_players
    bank = np.full((P, M), float(n0))
    busted = np.zeros((P, M), dtype=bool)
    p_col = p_grid[:, None]

    frames = {}
    frame_set = set(int(r) for r in frame_rounds)
    for t in range(1, n_rounds + 1):
        alive = ~busted
        step = np.where(rng.random((P, M)) < p_col, 1.0, -1.0)
        bank += step * alive
        newly = (bank <= 0) & alive
        busted |= newly
        bank[busted] = 0.0

        if t in frame_set:
            b = busted.mean(axis=1)
            profit = ((bank > n0) & ~busted).mean(axis=1)
            loss = 1.0 - b - profit
            frames[t] = (b, loss, profit)
    return frames


def theory_bust(p_grid, n0):
    """Asymptotic busted fraction: 1 for p<=1/2, (q/p)^N0 for p>1/2."""
    q = 1 - p_grid
    out = np.ones_like(p_grid)
    mask = p_grid > 0.5
    out[mask] = (q[mask] / p_grid[mask]) ** n0
    return out


def render(p_grid, frames, frame_rounds, n0, outdir):
    theory = theory_bust(p_grid, n0)
    fig, ax = plt.subplots(figsize=(11, 6))

    def draw(t):
        ax.clear()
        b, loss, profit = frames[t]
        ax.fill_between(p_grid, 0, b, color=C_BUST, label='busted')
        ax.fill_between(p_grid, b, b + loss, color=C_LOSS,
                        label='alive, below start')
        ax.fill_between(p_grid, b + loss, 1.0, color=C_PROFIT,
                        label='alive, in profit')
        ax.plot(p_grid, theory, 'k--', lw=1.3, alpha=0.7,
                label=r'theory $(q/p)^{N_0}$')
        ax.axvline(0.5, color='black', lw=1.0, ls=':', alpha=0.6)
        ax.set_xlim(p_grid[0], p_grid[-1])
        ax.set_ylim(0, 1)
        ax.set_xlabel('win probability $p$')
        ax.set_ylabel('population fraction')
        ax.set_title(f'Gambler\'s ruin: population vs $p$  '
                     f'(start $N_0$={n0}, fixed bet)')
        ax.text(0.97, 0.95, f'round = {t:,}', transform=ax.transAxes,
                ha='right', va='top', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.8))
        ax.legend(loc='center left', fontsize=9, framealpha=0.85)

    # Animation GIF (no static PNG on purpose); hold the last frame a bit
    seq = list(frame_rounds) + [frame_rounds[-1]] * 8
    anim = FuncAnimation(fig, draw, frames=seq, interval=180)
    gif_path = os.path.join(outdir, 'fig_ruin_population.gif')
    anim.save(gif_path, writer=PillowWriter(fps=6), dpi=DPI)
    plt.close()
    print('  saved fig_ruin_population.gif')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', default='./')
    ap.add_argument('--n-players', type=int, default=2000)
    ap.add_argument('--n0', type=int, default=20, help='start bankroll (units)')
    ap.add_argument('--n-p', type=int, default=65, help='resolution on p axis')
    ap.add_argument('--max-rounds', type=int, default=20000)
    ap.add_argument('--n-frames', type=int, default=48)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    p_grid = np.linspace(0.44, 0.60, args.n_p)
    # log-spaced round milestones (dense early where the action is)
    frame_rounds = np.unique(np.geomspace(
        1, args.max_rounds, args.n_frames).astype(int))

    print(f'Simulating: {args.n_p} p-values x {args.n_players} players, '
          f'N0={args.n0}, up to {args.max_rounds} rounds')
    frames = simulate(p_grid, args.n_players, args.n0, args.max_rounds,
                      frame_rounds, args.seed)
    print(f'  done, {len(frame_rounds)} frames')
    render(p_grid, frames, list(frame_rounds), args.n0, args.output_dir)
    print(f'figures in {os.path.abspath(args.output_dir)}')


if __name__ == '__main__':
    main()
