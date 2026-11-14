// Checks web/nested.html: seconds and minutes levels match index.html's seconds and
// minutes modes, hours light 5h of the face, and inner faces sit inside the 15-cell.
// Usage: node web/nested.test.js
const fs = require('fs');
const path = require('path');
const assert = require('assert');
const html = fs.readFileSync(path.join(__dirname, 'nested.html'), 'utf8');
const core = html.split('// --- core (shared with web/nested.test.js) ---')[1].split('// --- end core ---')[0];
const N = new Function(core + '; return { levelsAt, drawRects, perm };')();

const idx = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const ordering = eval('(' + idx.match(/let ordering = (\{[\s\S]*?\});/)[1] + ')');
const perm = new Function('ordering',
    idx.match(/function integerDigitsBasePad[\s\S]*?\n    }\n/)[0] +
    idx.match(/function perm\(t\)[\s\S]*?return ret;\n    }/)[0] + '; return perm;')(ordering);
const CP = 46221064723759104n;
// index.html with origin 1970 and no offset/P/N, for whole seconds t
function reference(t, tick, period) {
    t = BigInt(t);
    t = (t + period * tick * CP) % (60n * CP * period * tick);
    const k = (t / (period * tick)) % CP, ord = (t / tick) % period;
    return ordering[ord][perm(k)[ord]].slice().sort((a, b) => a - b);
}
const sorted = a => a.slice().sort((x, y) => x - y);
const sum = a => a.reduce((x, y) => x + y, 0);

let seed = 11;
const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
for (let i = 0; i < 3000; i++) {
    const k = BigInt(Math.floor(rnd() * 1e15)) * 46n + BigInt(i);
    assert.deepStrictEqual(N.perm(k), perm(k % CP));
    const t = Math.floor(rnd() * 4e9);
    const [h, m, s] = N.levelsAt(t, ['h', 'm', 's']);
    assert.deepStrictEqual(sorted(s.cells), reference(t, 1n, 60n));
    assert.deepStrictEqual(sorted(m.cells), reference(t, 60n, 60n));
    assert.strictEqual(s.value, t % 60);
    assert.strictEqual(m.value, Math.floor(t / 60) % 60);
    assert.strictEqual(h.value, Math.floor(t / 3600) % 12);
    assert.strictEqual(sum(h.cells) % 60, 5 * h.value);
}
// nested geometry: each inner face lies inside the 15-cell of the face around it
const st = N.levelsAt(123456, ['h', 'm', 's']);
const rects = N.drawRects(st);
const bgs = rects.filter(r => r.fill === '#1e1e1e');
assert.strictEqual(bgs.length, 3);
assert.strictEqual(rects.length, 3 + 6 + 6 + 7);
for (let i = 1; i < bgs.length; i++) {
    const o = bgs[i - 1], n = bgs[i], s = (i === 1 ? 1 : (bgs[i - 1].w) / 6);
    const fx = (i === 1 ? 0 : o.x), fy = (i === 1 ? 0 : o.y);
    assert(n.x >= fx && n.x + n.w <= fx + 3 * s + 1e-9, 'inner face inside 15-cell horizontally');
    assert(n.y >= fy + 5 * s && n.y + n.h <= fy + 10 * s + 1e-9, 'inner face inside 15-cell vertically');
}
console.log('nested: seconds/minutes match index.html, hours as 5h, geometry ok');
