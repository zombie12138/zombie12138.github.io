#!/usr/bin/env python3
"""
Four strategies side by side: population vs win-probability p, rolling over rounds.

Same idea as the fixed-bet ruin animation (x = p, stacked busted / alive-losing
/ alive-winning, animated as rounds advance), but 2x2 panels for fixed bet /
Kelly proportional / martingale / anti-martingale. Lets you compare how each
strategy's "cliff" forms (or doesn't) across p.

Output: fig_strategy_population.gif
Requirements: pip install numpy matplotlib pillow
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

B = 1.0
N0 = 20.0
M = 2000
ROUNDS = 8000
N_FRAMES = 44
SEED = 0

C_BUST, C_LOSS, C_PROFIT = '#e74c3c', '#f39c12', '#2ecc71'
STRATS = ['fixed', 'proportional', 'martingale', 'anti']
TITLES = {'fixed': 'Fixed bet', 'proportional': 'Proportional (Kelly $f^*$)',
          'martingale': 'Martingale', 'anti': 'Anti-martingale'}


def simulate(strat, p_grid, frame_rounds):
    rng = np.random.default_rng(SEED)
    P = len(p_grid)
    bank = np.full((P, M), N0)
    bet = np.ones((P, M))
    busted = np.zeros((P, M), bool)
    p_col = p_grid[:, None]
    f_star = np.maximum(0.0, (p_grid * B - (1 - p_grid)) / B)[:, None]  # Kelly per p
    want = set(int(r) for r in frame_rounds)
    out = {}
    for t in range(1, ROUNDS + 1):
        alive = ~busted
        wins = rng.random((P, M)) < p_col
        if strat == 'fixed':
            stake = np.ones((P, M))
        elif strat == 'proportional':
            stake = np.where(f_star > 0, np.clip(f_star * bank, 1.0, bank), 0.0)
        else:
            stake = np.minimum(bet, bank)
        bank = bank + np.where(wins, stake, -stake) * alive
        if strat == 'martingale':
            bet = np.where(wins, 1.0, bet * 2)
        elif strat == 'anti':
            bet = np.where(wins, bet * 2, 1.0)
        busted |= (bank < 1) & alive
        bank = np.where(busted, 0.0, bank)
        if t in want:
            b = busted.mean(axis=1)
            profit = ((bank > N0) & ~busted).mean(axis=1)
            out[t] = (b, 1 - b - profit, profit)
    return out


def main():
    outdir = os.path.dirname(os.path.abspath(__file__))
    p_grid = np.linspace(0.42, 0.60, 46)
    frame_rounds = np.unique(np.geomspace(1, ROUNDS, N_FRAMES).astype(int))
    print(f'{len(p_grid)} p-values x {M} players, {ROUNDS} rounds')
    data = {}
    for s in STRATS:
        data[s] = simulate(s, p_grid, frame_rounds)
        print(f'  {s} done')

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    def draw(i):
        t = frame_rounds[i]
        for ax, s in zip(axes, STRATS):
            ax.clear()
            b, loss, prof = data[s][t]
            ax.fill_between(p_grid, 0, b, color=C_BUST)
            ax.fill_between(p_grid, b, b + loss, color=C_LOSS)
            ax.fill_between(p_grid, b + loss, 1.0, color=C_PROFIT)
            ax.axvline(0.5, color='black', lw=0.9, ls=':', alpha=0.6)
            ax.set_xlim(p_grid[0], p_grid[-1])
            ax.set_ylim(0, 1)
            ax.set_title(TITLES[s], fontsize=10)
            ax.set_ylabel('fraction')
        axes[2].set_xlabel('win probability $p$')
        axes[3].set_xlabel('win probability $p$')
        fig.suptitle(f'Population vs $p$   |   round = {t:,}   '
                     f'(red=busted, orange=alive-losing, green=alive-winning)',
                     fontsize=12)

    seq = list(range(len(frame_rounds))) + [len(frame_rounds) - 1] * 8
    anim = FuncAnimation(fig, draw, frames=seq, interval=180)
    path = os.path.join(outdir, 'fig_strategy_population.gif')
    anim.save(path, writer=PillowWriter(fps=6), dpi=110)
    plt.close()
    print(f'  saved {path}')


if __name__ == '__main__':
    main()
