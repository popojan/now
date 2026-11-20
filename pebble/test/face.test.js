// Checks the watch core (src/embeddedjs/main.js) against web/custom.html on faces other than
// the default one: every mode, mod = each divisor of S, random origin/offset/P/N and times.
// Usage: TZ=Europe/Prague node test/face.test.js
const fs = require('fs');
const path = require('path');
const assert = require('assert');
const root = path.join(__dirname, '..', '..');
const src = fs.readFileSync(path.join(__dirname, '..', 'src', 'embeddedjs', 'main.js'), 'utf8');
const W = new Function(src.split('// --- core (shared with the node tests) ---')[1].split('// --- end core ---')[0] +
	'; return { clockState, cycleDigits, shownMask, parseConfig, bigStr, str, add, mul, divmod, big };')();
const html = fs.readFileSync(path.join(root, 'web', 'custom.html'), 'utf8');
const C = new Function(html.split('// --- core (shared with web/custom.test.js) ---')[1].split('// --- end core ---')[0] +
	'; return { buildOrdering, buildRadix, faceState, faceFromBits, mondrianColors };')();

let seed = 11;
const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
const digits = d => String(BigInt(Array.from({ length: d }, () => Math.floor(rnd() * 10)).join('')));

// big integers against BigInt
for (let i = 0; i < 3000; i++) {
	const a = (rnd() < 0.5 ? "-" : "") + digits(1 + Math.floor(rnd() * 30)), b = (rnd() < 0.5 ? "-" : "") + digits(1 + Math.floor(rnd() * 20));
	const m = 1 + Math.floor(rnd() * 5e7);
	const A = BigInt(a), B = BigInt(b), M = BigInt(m);
	assert.strictEqual(W.str(W.add(W.bigStr(a), W.bigStr(b))), String(A + B));
	assert.strictEqual(W.str(W.mul(W.bigStr(a), W.bigStr(b))), String(A * B));
	const { q, r } = W.divmod(W.bigStr(a), m);
	const R = ((A % M) + M) % M;
	assert.strictEqual(W.str(q), String((A - R) / M));
	assert.strictEqual(r, Number(R));
}
console.log('big integers ok');

const faces = [
	['1,2,4,6,9,12,20,30,36', 200 / 228],
	['1,2,4,6,12,16,30,49', 200 / 228],   // fastest digit on an odd sum
	['1,2,4,6,12,15,20,30', 200 / 228],   // S = 90
	['1,2,4,6,9,12,20,30,36', 1],         // Round 2
	['1,2,4,8,15,30,56,108,196', null],   // S = 420
];
const origins = ["1970-01-01T00:00:00Z", "2023-05-17T08:13:00Z", "-002000-01-01T00:00:00Z"];
let n = 0;
for (const [bits, shape] of faces) {
	const face = C.faceFromBits(bits, 1, shape);
	const S = face.W * face.H;
	const text = `${face.W}x${face.H};${face.cells.map(c => c.join('.')).join(',')};${C.mondrianColors(face.cells).join('')}`;
	const ordering = C.buildOrdering(face), radix = C.buildRadix(ordering);
	const divisors = [...Array(S).keys()].map(i => i + 1).filter(d => S % d === 0);
	for (let i = 0; i < 300; i++) {
		const s = {
			origin: origins[i % 3], P: String([1, 7, 49, 1000003, 11 * 13 * 17][i % 5]), N: digits(12),
			offset: [ "0", "5", "-3", "88000000000", digits(20)][(i >> 1) % 5],
			hours: ["auto", "12", "day", "24"][i % 4], mono: "0", face: text, mod: i % 4 ? String(divisors[i % divisors.length]) : ""
		};
		const cfg = W.parseConfig(s);
		const P = BigInt(s.P);
		const ctx = { S, ordering, radix, originMs: Date.parse(s.origin), offsetYears: BigInt(s.offset), P, N: BigInt(s.N) % P };
		const mod = BigInt(s.mod || (S % 60 === 0 ? 60 : S));
		const hour12 = (i >> 2) % 2 === 0, panel = s.hours === "auto" ? (hour12 ? "12" : "day") : s.hours;
		const modes = [[1n, 1n, mod], [3600n, BigInt(S), BigInt(S)], panel === "24" ? [3600n, 1n, 24n] : [panel === "day" ? 86400n : 43200n, BigInt(S), BigInt(S)]];
		const base = Date.parse("2026-11-20T12:00:00Z") + (rnd() - 0.5) * 4e11;
		for (let mode = 0; mode < 3; mode++) for (let d = 0; d < 3; d++) {
			const now = Math.floor(base + d * 37000);
			const local = new Date(now);
			const st = W.clockState(cfg.eng, now, cfg, mode, local, hour12);
			const mask = W.shownMask(cfg.eng, st, W.cycleDigits(cfg.eng, st, cfg));
			const wall = mode === 0 ? null : mode === 1 ? Math.floor((local.getMinutes() * 60 + local.getSeconds()) * S / 3600) : panel === "24" ? local.getHours()
				: Math.floor(((panel === "day" ? local.getHours() : local.getHours() % 12) * 3600 + local.getMinutes() * 60 + local.getSeconds()) * S / (panel === "day" ? 86400 : 43200));
			const want = C.faceState(ctx, now, ...modes[mode], wall).slice().sort((a, b) => a - b);
			const got = face.cells.map((c, j) => j).filter(j => mask & (1 << j));
			assert.deepStrictEqual(got, want, `${bits} ${JSON.stringify(s)} now=${now} mode=${mode}`);
			n++;
		}
	}
	console.log(`  ${bits} on ${face.W}x${face.H}: ok`);
}
console.log(`core vs web/custom.html (TZ=${process.env.TZ || "system"}): ${n} checks, 0 mismatches`);

// settings that must be refused
const ok = { origin: "1970-01-01T00:00:00Z", P: "1", N: "0", offset: "0", hours: "auto", mono: "0", face: "", mod: "" };
assert.throws(() => W.parseConfig({ ...ok, mod: "7" }), /divide/);
assert.throws(() => W.parseConfig({ ...ok, hours: "13" }), /hours/);
assert.throws(() => W.parseConfig({ ...ok, face: "2x2;2.0.0.2.1,2.0.1.2.1;00" }), /complete/);
assert.throws(() => W.parseConfig({ ...ok, face: "2x2;2.0.0.2.1,1.0.1.1.1;00" }), /gap/);
assert.strictEqual(W.parseConfig({ ...ok, mod: "30" }).mod, 30);
assert.deepStrictEqual(W.parseConfig({ ...ok, lit: "FFffFF", unlit: "555555" }).unlit, [85, 85, 85]);
assert.throws(() => W.parseConfig({ ...ok, lit: "red" }), /rrggbb/);
console.log('settings validation ok');

// the phone's copy of the tiling (src/pkjs/face.js) gives the faces used above
const F = require(path.join(__dirname, '..', 'src', 'pkjs', 'face.js'));
for (const [bits, shape] of [...faces, ['1,2,4,8,9,16,20,60', 200 / 228], ['20,15,12,6,4,2,1', 200 / 228], ['1,2,4,6,12,15,20', 1]]) {
	const face = C.faceFromBits(bits, 1, shape);
	const legacy = bits.split(',').map(Number).sort((a, b) => a - b).join() === '1,2,4,6,12,15,20';
	const want = legacy ? '' : `${face.W}x${face.H};${face.cells.map(c => c.join('.')).join(',')};${C.mondrianColors(face.cells).join('')}`;
	assert.strictEqual(F.faceText(bits, shape), want, bits);
}
console.log('src/pkjs/face.js matches custom.html');
// ... and stays a verbatim copy
{
	const fjs = fs.readFileSync(path.join(__dirname, '..', 'src', 'pkjs', 'face.js'), 'utf8');
	const fn = (text, name) => { const i = text.indexOf(`\nfunction ${name}(`); return text.slice(i, text.indexOf('\n}\n', i)); };
	for (const name of ['tile', 'faceFromBits', 'isLegacyLayout', 'adjacent', 'colorCells', 'mondrianColors'])
		assert.strictEqual(fn(fjs, name), fn(html, name).replace(/parseFace\(null, null\)/g, 'legacyFace()'), name);
}
console.log('src/pkjs/face.js is a verbatim copy');
