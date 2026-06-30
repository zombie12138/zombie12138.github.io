#!/usr/bin/env python3
"""
Kelly vs ruin: sorted-population profit map over bet fraction f, rolling over rounds.

x = bet fraction f, y = population percentile (the 2000 players sorted by profit),
color = that player's log-wealth (green profit, red loss/ruin). Animated as rounds
advance: the f* column goes mostly green, columns past 2f* turn red and sink to the
ruin floor.

Output: fig_kelly_population.gif
Requirements: pip install numpy matplotlib pillow
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter

P, B = 0.6, 1.0
M = 2000
ROUNDS = 400
N_FRAMES = 44
HOLD = 8                 # extra repeats of the final frame
SEED = 0
FLOOR = -6.0             # log-wealth ruin floor (lost ~99.75%)
F_STAR = (P * B - (1 - P)) / B          # 0.2
F_GRID = np.linspace(0.02, 0.95, 60)


def simulate(frame_rounds):
    rng = np.random.default_rng(SEED)
    F = len(F_GRID)
    logw = np.zeros((F, M))
    busted = np.zeros((F, M), bool)
    up = np.log1p(F_GRID * B)[:, None]
    dn = np.log1p(-F_GRID)[:, None]
    want = set(int(r) for r in frame_rounds)
    frames = {}
    for t in range(1, ROUNDS + 1):
        wins = rng.random((F, M)) < P
        logw = logw + np.where(wins, up, dn) * ~busted
        busted |= logw <= FLOOR
        logw = np.where(busted, FLOOR, logw)
        if t in want:
            frames[t] = np.sort(logw, axis=1).T      # (rank, f), low->high
    return frames


def main():
    outdir = os.path.dirname(os.path.abspath(__file__))
    frame_rounds = np.unique(np.geomspace(1, ROUNDS, N_FRAMES).astype(int))
    print(f'p={P}, f*={F_STAR}, {M} players x {len(F_GRID)} f-values, {ROUNDS} rounds')
    frames = simulate(frame_rounds)
    print('  simulated')

    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=FLOOR, vmax=8.0)
    fig, ax = plt.subplots(figsize=(11, 6))

    def draw(i):
        t = frame_rounds[i]
        ax.clear()
        im = ax.imshow(frames[t], origin='lower', aspect='auto', cmap='RdYlGn',
                       norm=norm, extent=[F_GRID[0], F_GRID[-1], 0, 100])
        ax.axvline(F_STAR, color='black', lw=1.2, ls='--', label=f'$f^*$={F_STAR:.1f}')
        ax.axvline(2 * F_STAR, color='black', lw=1.1, ls=':', label=f'$2f^*$={2*F_STAR:.1f}')
        ax.set_xlim(F_GRID[0], F_GRID[-1]); ax.set_ylim(0, 100)
        ax.set_xlabel('bet fraction $f$')
        ax.set_ylabel('population percentile (sorted by profit)')
        ax.set_title(f'Kelly vs ruin (p={P})   |   round = {t:,}   '
                     f'(green=profit, red=loss/ruin)')
        ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
        return [im]

    seq = list(range(len(frame_rounds))) + [len(frame_rounds) - 1] * HOLD
    anim = FuncAnimation(fig, draw, frames=seq, interval=180)
    path = os.path.join(outdir, 'fig_kelly_population.gif')
    anim.save(path, writer=PillowWriter(fps=6), dpi=115)
    plt.close()
    print(f'  saved {path}')


if __name__ == '__main__':
    main()
