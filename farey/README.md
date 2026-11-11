# Farey weight search

Tools for choosing cell weights like the clock's {1, 2, 4, 6, 12, 15, 20}.

- `farey_search.c` exhaustively finds n weights summing to S whose subset sums hit every fraction of Farey $F_k$. It does not require the other integers 0–S.
- `bits_search.py` enumerates *complete* sets: n weights summing to S = lcm(1..k) whose subset sums hit every integer 0–S. It ranks them by clock period, smoothness, how many weights divide S, and the cheapest tiling of a near-square grid.
- `tile.c` is the exact tiler behind `--tile`. It minimises the sum of |log aspect| over the cells and is compiled on first use.

```bash
gcc -O2 -o farey_search farey_search.c && ./farey_search 7 7 420
python3 bits_search.py 6 7 --distinct --divisors --tile 3 --sort tile --show 1   # picks the clock's weights
python3 bits_search.py 7 9 --distinct --tile 10 --sort tile --show 3 --svg f7.svg   # also draw them
python3 bits_search.py 7 9 --distinct --smooth 5 --tile 5 --sort tile --show 1
python3 bits_search.py 7 9 --distinct --tile 6 --sort tile --show 6 --no-cut --svg f7-mondrian.svg   # no straight cut through the whole face
```

`--show` also prints a link to `web/custom.html`, a live clock for that face. It takes the URL parameters of `index.html` (`origin`, `offset`, `P`, `N`, `period`, `mod`, `mono`), plus:
- `grid=WxH` and `cells=w.x.y.a.b,...`, the layout. Without them it shows the default face, and those subsets match `index.html` second by second.
- `cycle=60`, seconds per pass through all W·H states. With 420 states a state lasts 1/7 s, and the lit area is the elapsed share of the minute.
- `colors=hex,...` and `line=`, the colours and the line width in grid units.

`node web/custom.test.js` checks it against `index.html`.
