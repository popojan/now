// Checks web/nested2.html: seconds and minutes levels match index.html's seconds and
// minutes modes, hours light 5h of the face, and inner faces sit inside the 15-cell.
// Usage: node web/nested2.test.js
const fs = require('fs');
const path = require('path');
const assert = require('assert');
const html = fs.readFileSync(path.join(__dirname, 'nested2.html'), 'utf8');
const core = html.split('// --- core (shared with web/nested2.test.js) ---')[1].split('// --- end core ---')[0];
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
// constant lines: inner faces at exactly half scale; a dark 15-cell hides everything inside
const sumOf = cells => cells.reduce((x, y) => x + y, 0);
const g = 0.1, mk = cells => ({ cells });
const lit = [mk([15, 20, 1]), mk([15, 4]), mk([20])];
let r = N.drawRects(lit, g);
assert.strictEqual(r.length, 1 + 6 + 6 + 7);
const inner20 = r[1 + 6 + 5];  // level 2 cells go 1, 2, 4, 6, 12, 20: its 20 at (0, 5), half scale
assert.deepStrictEqual([inner20.x, inner20.y, inner20.w, inner20.h].map(v => +v.toFixed(9)), [g / 2, 5 + g / 2, 2 - g, 2.5 - g]);
r = N.drawRects([mk([20]), mk([15]), mk([1])], g);
assert.strictEqual(r.length, 1 + 6 + 1);  // outer 15 off: one white cell, nothing inside
assert(r.slice(1).every(c => c.fill !== '#1e1e1e'));
r = N.drawRects([mk([20]), mk([15]), mk([1])], g, 'lines');
assert.strictEqual(r.length, 1 + 6 + 6 + 7);
assert(r.slice(7).every(c => c.fill === '#ffffff'));  // inner levels drawn dark
console.log('nested2: seconds/minutes match index.html, hours as 5h, constant-line geometry and hiding ok');
