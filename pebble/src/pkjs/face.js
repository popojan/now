// Phone side of the face: weights -> tiling and colours, copied from web/custom.html's core
// (test/face.test.js checks they agree). The watch only draws the result.
// ES6 at most: the SDK bundles pkjs with webpack 1 (acorn 3).

const LEGACY = {
    grid: [6, 10],
    cells: [[20, 0, 0, 4, 5], [12, 4, 0, 2, 6], [15, 0, 5, 3, 5], [1, 3, 5, 1, 1],
            [2, 3, 6, 1, 2], [4, 4, 6, 2, 2], [6, 3, 8, 3, 2]],
    colors: {20: "#ffff00", 15: "#ff0000", 12: "#0000ff", 6: "#0000ff", 4: "#303030", 2: "#ffff00", 1: "#303030"},
};

const PALETTE = ["#ffff00", "#ff0000", "#0000ff", "#303030"];

// Port of farey/tile.c: exact tiling of W x H by one rectangle per weight, minimising
// the sum of |log aspect|; nocut rejects tilings a straight line splits in two.
// ratio makes R:1 (either way round) the ideal cell shape instead of a square.
// Returns {cost, cells} with cells [w, x, y, a, b] by decreasing w, or null.
function tile(weights, W, H, nocut, cap = Infinity, ratio = 1) {
    const logr = Math.log(ratio);
    if (W > 30) throw new Error("grid too wide to tile");
    const ws = weights.slice().sort((a, b) => b - a), n = ws.length, full = (1 << W) - 1;
    const opts = ws.map(w => {
        const o = [];
        for (let a = 1; a <= w; a++) if (w % a === 0 && a <= W && w / a <= H) o.push([a, w / a, Math.abs(Math.abs(Math.log(a / (w / a))) - logr)]);
        return o.sort((p, q) => p[2] - q[2]);  // stable: equal costs keep narrower first, as in tile.c
    });
    if (opts.some(o => !o.length)) return null;
    const lb = opts.map(o => o[0][2]);
    const oa = opts.map(o => Int32Array.from(o, v => v[0])), ob = opts.map(o => Int32Array.from(o, v => v[1]));
    const oc = opts.map(o => Float64Array.from(o, v => v[2]));
    const rows = new Int32Array(H), used = new Uint8Array(n);
    const px = new Int32Array(n), py = new Int32Array(n), pa = new Int32Array(n), pb = new Int32Array(n);
    let best = cap, bestp = null;
    const crossed = (lo, len, at) => {
        for (let i = 0; i < n; i++) if (used[i] && lo[i] < at && at < lo[i] + len[i]) return true;
        return false;
    };
    const rec = (cost, rest) => {
        if (cost + rest >= best - 1e-12) return;
        let y = 0;
        while (y < H && rows[y] === full) y++;
        if (nocut && y > 0 && y < H && !crossed(py, pb, y)) return;
        if (y === H) {
            if (nocut) for (let x = 1; x < W; x++) if (!crossed(px, pa, x)) return;
            best = cost;
            bestp = ws.map((w, i) => [w, px[i], py[i], pa[i], pb[i]]);
            return;
        }
        const free = ~rows[y] & full, x = 31 - Math.clz32(free & -free);
        let run = 0;
        while (x + run < W && !(rows[y] >> (x + run) & 1)) run++;
        for (let i = 0; i < n; i++) {
            if (used[i] || (i > 0 && ws[i] === ws[i - 1] && !used[i - 1])) continue;
            const A = oa[i], B = ob[i], Cc = oc[i];
            for (let o = 0; o < A.length; o++) {
                const a = A[o], b = B[o];
                if (a > run || y + b > H) continue;
                const m = ((1 << a) - 1) << x;
                let ok = true;
                for (let r = y; r < y + b; r++) if (rows[r] & m) { ok = false; break; }
                if (!ok) continue;
                for (let r = y; r < y + b; r++) rows[r] |= m;
                used[i] = 1; px[i] = x; py[i] = y; pa[i] = a; pb[i] = b;
                rec(cost + Cc[o], rest - lb[i]);
                used[i] = 0;
                for (let r = y; r < y + b; r++) rows[r] &= ~m;
            }
        }
    };
    rec(0, lb.reduce((s, v) => s + v, 0));
    return bestp && { cost: best, cells: bestp };
}

const LEGACY_WEIGHTS = LEGACY.cells.map(c => c[0]).sort((a, b) => a - b).join();
const sortedWeights = cells => cells.map(c => c[0]).sort((a, b) => a - b).join();
const transpose = face => ({ W: face.H, H: face.W, cells: face.cells.map(([w, x, y, a, b]) => [w, y, x, b, a]) });

// Face from weights alone: the cheapest tiling, with the Mondrian rule unless no
// such tiling exists.  Without shape, any W x H = sum with H/W <= 2 (portrait);
// with shape (width/height), the grid nearest that shape, landscape included, and
// the next nearest only if nothing fits.  The default weights keep index.html's
// layout (turned on its side for a landscape grid) while ratio is 1.
function faceFromBits(bits, ratio = 1, shape = null) {
    const ws = bits.split(",").map(Number);
    if (ws.some(w => !Number.isInteger(w) || w < 1)) throw new Error("bad bits " + bits);
    const S = ws.reduce((a, b) => a + b, 0);
    let reach = 0;
    for (const w of ws.slice().sort((a, b) => a - b)) {
        if (w > reach + 1) throw new Error(`no subset of the weights sums to ${reach + 1}: not complete`);
        reach += w;
    }
    const grids = [];
    for (let W = 1; W <= S; W++) if (S % W === 0) grids.push([W, S / W]);
    const legacy = ratio === 1 && ws.slice().sort((a, b) => a - b).join() === LEGACY_WEIGHTS;
    const best = (cands, nocut) => {
        let b = null;
        for (const [W, H] of cands) {  // tile portrait, turn afterwards
            const t = tile(ws, Math.min(W, H), Math.max(W, H), nocut, b ? b.cost : Infinity, ratio);
            if (t) b = { cost: t.cost, face: W <= H ? { W, H, cells: t.cells } : transpose({ W: H, H: W, cells: t.cells }) };
        }
        return b && b.face;
    };
    if (!shape) {
        if (legacy) return legacyFace();
        // squarest first: an early bound prunes the rest
        const cands = grids.filter(([W, H]) => W <= H && H <= 2 * W).reverse();
        const f = best(cands, true) || best(cands, false);
        if (f) return f;
        throw new Error(`no rectangle W x H = ${S} with H/W <= 2 fits these cells; try shape= or pass grid= and cells=`);
    }
    const off = ([W, H]) => Math.abs(Math.log(W / H / shape));
    // equally far (1:1 between 20x21 and 21x20): portrait first
    grids.sort((p, q) => (Math.abs(off(p) - off(q)) > 1e-9 ? off(p) - off(q) : 0) || (p[0] - q[0]));
    for (const g of grids) {
        if (legacy && g[0] === 6) return legacyFace();
        if (legacy && g[0] === 10) return transpose(legacyFace());
        const f = best([g], true) || best([g], false);
        if (f) return f;
    }
    throw new Error("no grid fits these cells");
}

function isLegacyLayout(face) {
    const same = f => f.W === 6 && f.H === 10 && JSON.stringify(f.cells) === JSON.stringify(LEGACY.cells);
    return same(face) || same(transpose(face));
}

// cells sharing a stretch of edge (corners don't count)
function adjacent([, x1, y1, a1, b1], [, x2, y2, a2, b2]) {
    const overlap = (p, l, q, m) => Math.min(p + l, q + m) - Math.max(p, q) > 0;
    return ((x1 + a1 === x2 || x2 + a2 === x1) && overlap(y1, b1, y2, b2)) ||
           ((y1 + b1 === y2 || y2 + b2 === y1) && overlap(x1, a1, x2, a2));
}

// Palette index per cell, no two neighbours alike (a rectangle dissection is planar,
// so 4 colours always suffice).  Largest cells first, each preferring the colour
// the plain rotation would give it; `fixed` maps cell index to a preset colour.
function colorCells(cells, k, fixed = {}) {
    const order = cells.map((c, i) => i).filter(i => !(i in fixed)).sort((i, j) => cells[j][0] - cells[i][0]);
    const col = cells.map((c, i) => i in fixed ? fixed[i] : -1);
    const rec = r => {
        if (r === order.length) return true;
        const i = order[r];
        for (let d = 0; d < k; d++) {
            const c = (r + d) % k;
            if (cells.some((o, j) => col[j] === c && adjacent(cells[i], o))) continue;
            col[i] = c;
            if (rec(r + 1)) return true;
        }
        col[i] = -1;
        return false;
    };
    if (!rec(0)) throw new Error(`no ${k}-colouring without equal neighbours`);
    return col;
}

// the 1x1 cell is black, as on the original face (unless 1-cells touch each other)
function mondrianColors(cells) {
    const black = {};
    cells.forEach((c, i) => { if (c[0] === 1) black[i] = PALETTE.indexOf("#303030"); });
    try { return colorCells(cells, PALETTE.length, black); } catch (e) { return colorCells(cells, PALETTE.length); }
}

function legacyFace() {
    return { W: LEGACY.grid[0], H: LEGACY.grid[1], cells: LEGACY.cells.map(c => c.slice()) };
}

// "WxH;w.x.y.a.b,...;colours" for the watch (main.js parseFace); "" keeps its built-in face.
// shape: width / height of the display.
function faceText(bits, shape) {
    if (!bits || sortedWeights(bits.split(",").map(w => [+w])) === LEGACY_WEIGHTS) return "";
    const face = faceFromBits(bits, 1, shape);
    const colors = isLegacyLayout(face) ? face.cells.map(c => PALETTE.indexOf(LEGACY.colors[c[0]])) : mondrianColors(face.cells);
    return face.W + "x" + face.H + ";" + face.cells.map(c => c.join(".")).join(",") + ";" + colors.join("");
}

module.exports = { faceText, faceFromBits, mondrianColors };
