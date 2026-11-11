#!/usr/bin/env python3
"""Rank complete weight sets: n weights summing to S = lcm(1..k) whose subset
sums hit every integer 0..S (so Farey F_k comes for free).

usage: bits_search.py k n [--distinct] [--divisors] [--tile N] [--top N] [--sort KEY] [--show N]
"""
import argparse
import os
import subprocess
from math import lcm, log, log2, prod

def complete_sets(S, n):
    # sorted w_1..w_n is complete iff each w_i <= 1 + sum of the previous ones
    out = []
    def dfs(d, lo, tot, ws):
        if d == n - 1:
            w = S - tot
            if lo <= w <= tot + 1:
                out.append(ws + [w])
            return
        for w in range(lo, min(tot + 1, (S - tot) // (n - d)) + 1):
            dfs(d + 1, w, tot + w, ws + [w])
    dfs(0, 1, 0, [])
    return out

def reps(ws, S):
    c = [1] + [0] * S
    for w in ws:
        for s in range(S, w - 1, -1):
            c[s] += c[s - w]
    return c

def gpf(x):
    p, best = 2, 1
    while p * p <= x:
        while x % p == 0:
            best, x = p, x // p
        p += 1
    return max(best, x) if x > 1 else best

def dims(w):
    return [(a, w // a) for a in range(1, w + 1) if w % a == 0]

def aspect_cost(a, b):
    return abs(log(a / b))

TILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tile")

def best_tiling(ws, W, H, cap=None):
    """Cheapest exact tiling of W x H (sum of |log aspect| over cells), via ./tile."""
    if not os.path.exists(TILE):
        subprocess.run(["gcc", "-O2", "-o", TILE, TILE + ".c", "-lm"], check=True)
    env = dict(os.environ, **({"TILE_MAX_COST": str(cap)} if cap is not None else {}))
    out = subprocess.run([TILE, str(W), str(H), *map(str, ws)], capture_output=True, text=True, env=env).stdout.split("\n")
    if out[0] == "none":
        return None
    return float(out[0]), [tuple(map(int, l.split())) for l in out[1:] if l]

def grids(S, max_ratio):
    return [(W, S // W) for W in range(1, S + 1) if S % W == 0 and W <= S // W and S // W / W <= max_ratio]

def draw(place, W, H):
    g = [["?"] * W for _ in range(H)]
    sym = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    legend = []
    for j, (w, x, y, a, b) in enumerate(sorted(place, key=lambda p: (p[2], p[1]))):
        for r in range(y, y + b):
            for c in range(x, x + a):
                g[r][c] = sym[j]
        legend.append(f"{sym[j]}={w}({a}x{b})")
    return "\n".join("    " + " ".join(r) for r in g) + "\n    " + " ".join(legend)

MONDRIAN = ["#d40920", "#1356a2", "#f7d842", "#f2f2f2"]

def svg(tilings, size=320, cols=3):
    """tilings: list of (title, place, W, H); returns an SVG string, one panel per tiling."""
    pad, cap = 20, 40
    panels = []
    for i, (title, place, W, H) in enumerate(tilings):
        u = size / max(W, H)
        ox = pad + (i % cols) * (size + pad)
        oy = pad + (i // cols) * (size + pad + cap)
        p = [f'<g transform="translate({ox},{oy})">']
        for j, (w, x, y, a, b) in enumerate(sorted(place, key=lambda q: -q[0])):
            p.append(f'<rect x="{x*u:.1f}" y="{y*u:.1f}" width="{a*u:.1f}" height="{b*u:.1f}" '
                     f'fill="{MONDRIAN[j % len(MONDRIAN)]}"/>')
        # faint unit grid so cell dimensions can be read off
        for c in range(1, W):
            p.append(f'<line x1="{c*u:.1f}" y1="0" x2="{c*u:.1f}" y2="{H*u:.1f}" stroke="#000" stroke-opacity=".12" stroke-width=".5"/>')
        for r in range(1, H):
            p.append(f'<line x1="0" y1="{r*u:.1f}" x2="{W*u:.1f}" y2="{r*u:.1f}" stroke="#000" stroke-opacity=".12" stroke-width=".5"/>')
        for w, x, y, a, b in place:
            p.append(f'<rect x="{x*u:.1f}" y="{y*u:.1f}" width="{a*u:.1f}" height="{b*u:.1f}" '
                     f'fill="none" stroke="#111" stroke-width="{max(1.5, u/5):.1f}"/>')
            # shrink the label to the cell width (~0.6 em per character)
            label = f"{w} ({a}×{b})"
            fs = min(22, b * u * 0.45, a * u * 0.85 / (0.6 * len(label)))
            if fs < 9:
                label = f"{w}"
                fs = min(22, b * u * 0.45, a * u * 0.85 / (0.6 * len(label)))
            if fs >= 5:
                p.append(f'<text x="{(x+a/2)*u:.1f}" y="{(y+b/2)*u:.1f}" font-size="{fs:.1f}" '
                         f'text-anchor="middle" dominant-baseline="central">{label}</text>')
        p.append(f'<text x="0" y="{H*u + 18:.1f}" font-size="13">{title}</text>')
        p.append(f'<text x="0" y="{H*u + 34:.1f}" font-size="11" fill="#555">{W}×{H} grid</text>')
        p.append("</g>")
        panels.append("\n".join(p))
    n = len(tilings)
    width = pad + min(n, cols) * (size + pad)
    height = pad + ((n + cols - 1) // cols) * (size + pad + cap)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'font-family="sans-serif"><rect width="100%" height="100%" fill="#fff"/>\n'
            + "\n".join(panels) + "\n</svg>\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("k", type=int)
    ap.add_argument("n", type=int)
    ap.add_argument("--distinct", action="store_true", help="no repeated weights")
    ap.add_argument("--divisors", action="store_true", help="every weight divides S (cells are unit fractions)")
    ap.add_argument("--smooth", type=int, default=0, help="largest allowed prime factor of any weight")
    ap.add_argument("--tile", type=int, default=0, help="find the N sets with the cheapest tiling (exact)")
    ap.add_argument("--max-grid-ratio", type=float, default=2.0)
    ap.add_argument("--sort", default="period", choices=["period", "balance", "tile", "gpf", "div"])
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--show", type=int, default=0, help="draw the best N tilings")
    ap.add_argument("--svg", help="also render the --show tilings to this SVG file")
    args = ap.parse_args()
    S = lcm(*range(1, args.k + 1))
    sets = complete_sets(S, args.n)
    print(f"S={S}, n={args.n}: {len(sets)} complete multisets", end="")
    if args.distinct:
        sets = [s for s in sets if len(set(s)) == len(s)]
    if args.divisors:
        sets = [s for s in sets if all(S % w == 0 for w in s)]
    if args.smooth:
        sets = [s for s in sets if all(gpf(w) <= args.smooth for w in s)]
    print(f", {len(sets)} after filters")
    rows = []
    for ws in sets:
        c = reps(ws, S)
        c[0] += c[S]  # second 0 is shown as empty or full (sum % S)
        rows.append({
            "ws": ws,
            "period": sum(log2(x) for x in c[:S]),
            "gpf": max(gpf(w) for w in ws),
            "div": sum(S % w == 0 for w in ws),
            # cost of the squarest rectangle per cell, ignoring whether they tile
            "balance": sum(min(aspect_cost(a, b) for a, b in dims(w)) for w in ws),
            "tile": None,
        })
    if args.tile:
        # balance is a lower bound on tiling cost, so stop once it can't beat the N-th best
        rows.sort(key=lambda r: r["balance"])
        found = []
        for r in rows:
            cap = found[args.tile - 1] if len(found) >= args.tile else None
            if cap is not None and r["balance"] >= cap:
                break
            best = None
            for W, H in grids(S, args.max_grid_ratio):
                c = min(x for x in (cap, best[0] if best else None) if x is not None) if (cap or best) else None
                t = best_tiling(r["ws"], W, H, c)
                if t and (best is None or t[0] < best[0]):
                    best = (t[0], W, H, t[1])
            r["tile"] = best
            if best:
                found = sorted(found + [best[0]])[:args.tile]
    key = {"period": lambda r: -r["period"], "gpf": lambda r: (r["gpf"], -r["period"]),
           "balance": lambda r: r["balance"],
           "div": lambda r: (-r["div"], r["tile"][0] if r["tile"] else r["balance"]),
           "tile": lambda r: r["tile"][0] if r["tile"] else float("inf")}[args.sort]
    rows.sort(key=key)
    for r in rows[:args.top]:
        t = f"tile {r['tile'][0]:.2f} on {r['tile'][1]}x{r['tile'][2]}" if r["tile"] else ""
        print(f"  {r['ws']}  period 2^{r['period']:.2f}  gpf {r['gpf']}  div {r['div']}/{len(r['ws'])}  balance {r['balance']:.2f}  {t}")
    shown = [r for r in rows[:args.show] if r["tile"]]
    for r in shown:
        print(f"\n  {r['ws']}")
        print(draw(r["tile"][3], r["tile"][1], r["tile"][2]))
    if args.svg:
        with open(args.svg, "w") as f:
            f.write(svg([(", ".join(map(str, r["ws"])) + f"  (cost {r['tile'][0]:.2f})", r["tile"][3], r["tile"][1], r["tile"][2])
                         for r in shown]))
        print(f"\nwrote {args.svg} ({len(shown)} tilings)")

if __name__ == "__main__":
    main()
