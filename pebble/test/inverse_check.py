"""Check minutes dumped by core.test.js against inverse/clock_inverse.py (perm and inverse_perm)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "inverse"))
from clock_inverse import ORDERING, PERIOD, inverse_perm, perm  # noqa: E402

CELLS = [1, 2, 4, 6, 12, 15, 20]
dump = json.load(open(sys.argv[1]))
bad = 0
for k, masks in dump:
    k = int(k)
    idx = perm(k)
    expected = [sum(1 << CELLS.index(c) for c in ORDERING[s][idx[s]]) for s in range(60)]
    bad += expected != masks
    bad += inverse_perm(idx) != k % PERIOD
print(f"core vs inverse/clock_inverse.py: {len(dump)} minutes, {bad} mismatches")
sys.exit(1 if bad else 0)
