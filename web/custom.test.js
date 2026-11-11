// Checks web/custom.html against web/index.html on the default face (all three
// modes, origin/offset/P/N), then sanity-checks a 420-state face.
// Usage: node web/custom.test.js
const fs = require('fs');
const path = require('path');
const assert = require('assert');
const html = fs.readFileSync(path.join(__dirname, 'custom.html'), 'utf8');
const core = html.split('// --- core (shared with web/custom.test.js) ---')[1].split('// --- end core ---')[0];
const C = new Function(core + '; return { parseFace, buildOrdering, buildRadix, choices, faceState, colorCells, mondrianColors, adjacent, LEGACY };')();

const idx = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const ordering = eval('(' + idx.match(/let ordering = (\{[\s\S]*?\});/)[1] + ')');
const perm = new Function('ordering',
    idx.match(/function integerDigitsBasePad[\s\S]*?\n    }\n/)[0] +
    idx.match(/function perm\(t\)[\s\S]*?return ret;\n    }/)[0] + '; return perm;')(ordering);
const CP = 46221064723759104n;

// index.html's setInterval body
function reference(now, origin, offset, P, N, tick, period) {
    let t = BigInt(Math.floor((now - origin) / 1000.0));
    t = (t + period * tick * CP + offset * 365n * 24n * 3600n) % (60n * CP * period * tick);
    const k = (t / (period * tick) * P + N % P) % CP;
    const ord = (t / tick) % period;
    return ordering[ord][perm(k)[ord]].slice().sort((a, b) => a - b);
}

let seed = 7;
const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
const big = d => BigInt(Math.floor(rnd() * 10 ** d));

function context(face, origin, offset, P, N) {
    const ord = C.buildOrdering(face);
    return { S: face.W * face.H, ordering: ord, radix: C.buildRadix(ord), originMs: origin, offsetYears: offset, P, N: N % P };
}

const face = C.parseFace(null, null);
assert.strictEqual(C.buildRadix(C.buildOrdering(face)).period, CP);
const modes = [[1n, 1n, 60n, 1n, 60n], [60n, 1n, 60n, 60n, 60n], [3600n, 1n, 24n, 3600n, 24n]];
let n = 0;
for (let i = 0; i < 4000; i++) {
    const origin = Date.UTC(1970, 0, 1) + Math.floor(rnd() * 2e12);
    const now = origin + Math.floor((rnd() - 0.05) * 3e12);
    const offset = rnd() < 0.5 ? 0n : big(3);
    const P = rnd() < 0.5 ? 1n : 1n + 2n * big(4) * 3n + 4n;  // odd and not divisible by 3 half the time
    const N = big(5);
    const ctx = context(face, origin, offset, P, N);
    for (const [num, den, mod, tick, period] of modes) {
        const got = C.faceState(ctx, now, num, den, mod).map(i => face.cells[i][0]).sort((a, b) => a - b);
        assert.deepStrictEqual(got, reference(now, origin, offset, P, N, tick, period), `now=${now} origin=${origin}`);
        n++;
    }
}
console.log(`default face matches index.html in ${n} cases`);

// 420-state face from farey/bits_search.py --no-cut
const f = C.parseFace('20x21', '196.0.0.14.14,108.14.0.6.18,56.0.14.8.7,30.8.14.6.5,15.15.18.5.3,8.8.19.4.2,4.13.19.2.2,2.12.19.1.2,1.14.18.1.1');
const ctx = context(f, 0, 0n, 1n, 0n);
const counts = ctx.ordering.map(o => o.length);
assert.strictEqual(ctx.radix.period, counts.reduce((a, m) => a * BigInt(m), 1n));
assert.strictEqual(ctx.radix.period, 2n ** 88n * 9n);
// one state every 1/7 s; each state shows cells summing to its index
for (let s = 0; s < 420; s++) {
    const ms = Math.ceil(s * 60000 / 420);
    const sub = C.faceState(ctx, ms, 60n, 420n, 420n);
    assert.strictEqual(sub.reduce((a, i) => a + f.cells[i][0], 0) % 420, s);
}
// minute mode on 420 states shows minute m as 7m: the same share of the face
for (let m = 0; m < 60; m++) {
    const sub = C.faceState(ctx, m * 60000, 60n, 1n, 60n);
    assert.strictEqual(sub.reduce((a, i) => a + f.cells[i][0], 0) % 420, 7 * m);
}
// consecutive minutes differ (k decodes to different choices)
const a = C.choices(ctx.radix, 12345n, 420).join(), b = C.choices(ctx.radix, 12346n, 420).join();
assert.notStrictEqual(a, b);
for (const bad of [['6x10', '20.0.0.4.5'], ['2x2', '2.0.0.2.1,2.0.0.2.1'], ['3x1', '3.0.0.3.1']]) {
    assert.throws(() => C.buildOrdering(C.parseFace(...bad)));
}
// no two neighbouring cells share a colour, the default face's hand-picked colours included
const noClash = (cells, col) => cells.every((c, i) => cells.every((o, j) => j <= i || !C.adjacent(c, o) || col[i] !== col[j]));
const fc = C.mondrianColors(f.cells);
assert(noClash(f.cells, fc));
assert.strictEqual(fc[f.cells.findIndex(c => c[0] === 1)], 3);  // 1x1 is black
assert(noClash(C.LEGACY.cells, C.LEGACY.cells.map(c => C.LEGACY.colors[c[0]])));
assert(C.adjacent([1, 0, 0, 1, 1], [1, 1, 0, 1, 1]) && !C.adjacent([1, 0, 0, 1, 1], [1, 1, 1, 1, 1]));
console.log('420-state face: period 2^88*3^2, 1/7 s states, minute mode, validation and colouring ok');
