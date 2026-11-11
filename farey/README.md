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
- `bits=1,2,4,8,15,30,56,108,196`, the weights alone. The page tiles them itself, the same way as `--tile --no-cut`: it picks the cheapest tiling with no straight cut through the face, and allows cuts only if no other tiling exists. The default weights keep `index.html`'s layout.
- `grid=WxH` and `cells=w.x.y.a.b,...`, an explicit layout instead. With neither, the page shows the default face, and its subsets match `index.html` second by second.
- `speed=1`, states per second. With 420 states the default cycle lasts 7 minutes. `speed=7` runs one cycle per minute, so a state lasts 1/7 s and the lit area is the elapsed share of the minute.
- `colors=hex,...` and `line=`, the colours and the line width in grid units.

`node web/custom.test.js` checks it against `index.html`.
