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
python3 bits_search.py 5 9 --sum 120 --distinct --tile 5 --sort tile --no-cut --max-grid-ratio 1.25 --show 2   # S=120 (2-minute cycle) on 10x12
```

`--show` also prints a link to `web/custom.html`, a live clock for that face. It takes the URL parameters of `index.html` (`origin`, `offset`, `P`, `N`, `period`, `mod`, `mono`), plus:
- `bits=1,2,4,8,15,30,56,108,196`, the weights alone. The page tiles them itself, the same way as `--tile --no-cut`: it picks the cheapest tiling with no straight cut through the face, and allows cuts only if no other tiling exists. The default weights keep `index.html`'s layout.
- `shape=16:9`, width:height of the whole face. The page uses the grid nearest that shape, landscape included, then stretches the units so the shape is exact. A uniform stretch keeps the areas proportional to the weights. With only `shape=`, the default face turns on its side when landscape is nearer, and it still matches `index.html`. Wide shapes can take a second or two to tile.
- `ratio=1.618` or `ratio=3:2`, the ideal cell shape for `bits=`, either way round. The default is 1, square cells. The same option in the tools is `bits_search.py --ratio` and `tile -r`.
- `grid=WxH` and `cells=w.x.y.a.b,...`, an explicit layout instead. With neither, the page shows the default face, and its subsets match `index.html` second by second.
- `speed=1`, states per second. A full face is one minute whenever S is a multiple of 60 × speed: with S = 120 second s lights 2s, with S = 420 every 7th sum is shown, and `speed=7` uses all 420 at 1/7 s each. Other S cycle through all states (S = 90: 90 s); `mod=` sets the states per cycle explicitly.
- `colors=hex,...` and `line=`, the colours and the line width in grid units.
- Click cycles seconds, minutes and hours. The hour panel is a 12-hour dial: the lit area is the elapsed share of 12 h, in steps of 12 h / S (8 minutes for S = 90). `hours=day` is a 24-hour dial (share of the day), `hours=24` shows the hour 0–23 as a sum instead. The minute and hour panels follow local time, `tz=utc` shows UTC.

`node web/custom.test.js` checks it against `index.html`.
