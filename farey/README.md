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
```
