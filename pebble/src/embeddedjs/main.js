import Poco from "commodetto/Poco";
import Message from "pebble/message";
import Accelerometer from "embedded:sensor/Accelerometer";

// --- core (shared with the node tests) ---
// Any complete face as in web/custom.html: cells [w, x, y, a, b] on a W x H grid, the subsets
// summing to s, and one mixed-radix digit of k = (cycle * P + N) per shown sum. XS on Pebble
// strips BigInt ("dead strip"), so big integers are base-1e7 limbs.

// Default face in the order of hraj.si's bitmasks (bit i = cell i); colours index PALETTE
const LEGACY = {
	W: 6, H: 10,
	cells: [[1, 3, 5, 1, 1], [2, 3, 6, 1, 2], [4, 4, 6, 2, 2], [6, 3, 8, 3, 2], [12, 4, 0, 2, 6], [15, 0, 5, 3, 5], [20, 0, 0, 4, 5]],
	colors: [3, 0, 3, 2, 2, 1, 0]
};
// ordering from hraj.si/now as bitmasks; ORDERING[s] lists combos summing to s
const ORDERING = [
	[127,0],[1],[2],[3],[4],[5],[8,6],[9,7],[10],[11],
	[12],[13],[16,14],[17,15],[18],[32,19],[33,20],[34,21],[35,24,22],[36,25,23],
	[64,37,26],[65,40,38,27],[66,41,39,28],[67,42,29],[68,43,30],[69,44,31],[72,70,45],[73,71,48,46],[49,74,47],[75,50],
	[51,76],[77,52],[80,53,78],[81,79,56,54],[82,57,55],[96,83,58],[97,84,59],[98,85,60],[99,88,86,61],[100,89,87,62],
	[101,90,63],[104,102,91],[105,103,92],[106,93],[107,94],[108,95],[109],[112,110],[113,111],[114],
	[115],[116],[117],[120,118],[121,119],[122],[123],[124],[125],[126]
];

// --- signed big integers { s: 1 | -1, d: base-1e7 limbs, least significant first } ---
const BASE = 10000000;
const trim = d => { while (d.length && !d[d.length - 1]) d.pop(); return d; };
const norm = (s, d) => ({ s: trim(d).length ? s : 1, d });
function big(x) { // integer, |x| < 2^53
	const d = [];
	for (let v = Math.abs(x); v > 0; v = Math.floor(v / BASE)) d.push(v % BASE);
	return norm(x < 0 ? -1 : 1, d);
}
function bigStr(str) { // decimal, optional sign
	const digits = str.replace(/^[+-]/, ""), d = [];
	for (let e = digits.length; e > 0; e -= 7) d.push(Number(digits.slice(Math.max(0, e - 7), e)));
	return norm(str[0] === "-" ? -1 : 1, d);
}
function str(x) {
	if (!x.d.length) return "0";
	let r = String(x.d[x.d.length - 1]);
	for (let i = x.d.length - 2; i >= 0; i--) r += String(x.d[i]).padStart(7, "0");
	return (x.s < 0 ? "-" : "") + r;
}
function cmpMag(a, b) {
	if (a.length !== b.length) return a.length - b.length;
	for (let i = a.length - 1; i >= 0; i--) if (a[i] !== b[i]) return a[i] - b[i];
	return 0;
}
function addMag(a, b) {
	const r = [];
	let c = 0;
	for (let i = 0; i < Math.max(a.length, b.length) || c; i++) {
		const t = (a[i] || 0) + (b[i] || 0) + c;
		r.push(t % BASE);
		c = t >= BASE ? 1 : 0;
	}
	return r;
}
function subMag(a, b) { // |a| >= |b|
	const r = [];
	let c = 0;
	for (let i = 0; i < a.length; i++) {
		let t = a[i] - (b[i] || 0) - c;
		c = t < 0 ? 1 : 0;
		r.push(t + c * BASE);
	}
	return r;
}
function add(x, y) {
	if (x.s === y.s) return norm(x.s, addMag(x.d, y.d));
	return cmpMag(x.d, y.d) >= 0 ? norm(x.s, subMag(x.d, y.d)) : norm(y.s, subMag(y.d, x.d));
}
function mul(x, y) {
	const r = new Array(x.d.length + y.d.length).fill(0);
	for (let i = 0; i < x.d.length; i++) {
		let c = 0;
		for (let j = 0; j < y.d.length || c; j++) {
			const t = r[i + j] + x.d[i] * (y.d[j] || 0) + c;
			r[i + j] = t % BASE;
			c = Math.floor(t / BASE);
		}
	}
	return norm(x.s * y.s, r);
}
// floor division by 1 <= m < 9e8: x = q * m + r, 0 <= r < m
function divmod(x, m) {
	const q = new Array(x.d.length);
	let r = 0;
	for (let i = x.d.length - 1; i >= 0; i--) {
		const t = r * BASE + x.d[i];
		q[i] = Math.floor(t / m);
		r = t % m;
	}
	if (x.s > 0 || !r) return { q: norm(x.s, q), r };
	return { q: norm(-1, addMag(trim(q), [1])), r: m - r };
}

// --- faces ---
// The tables are typed arrays: the XS heap has room for about 1500 slots.
// Ordering: masks (bit i = cell i) of the subsets summing to s mod S are
// masks[off[s] .. off[s + 1]), full face first at 0. Enumeration bit j is the
// j-th smallest weight, as in custom.html.
function buildOrdering(face) {
	const S = face.W * face.H, ws = face.cells.map(c => c[0]), n = ws.length;
	const off = new Uint16Array(S + 1);
	let masks;
	if (ws.slice().sort((a, b) => a - b).join() === "1,2,4,6,12,15,20") {
		const idx = LEGACY.cells.map(c => ws.indexOf(c[0]));
		masks = new Uint16Array(128);
		let k = 0;
		ORDERING.forEach((opts, sum) => {
			off[sum] = k;
			for (const m of opts) {
				let r = 0;
				for (let i = 0; i < 7; i++) if (m >> i & 1) r |= 1 << idx[i];
				masks[k++] = r;
			}
		});
		off[S] = k;
		return { off, masks, legacy: true };
	}
	const order = ws.map((w, i) => i).sort((i, j) => (ws[i] - ws[j]) || (i - j));
	const total = 1 << n, sums = new Uint16Array(total), cellMask = new Uint16Array(total);
	for (let m = 0; m < total - 1; m++) {
		let sum = 0, mask = 0;
		for (let j = 0; j < n; j++) if (m >> j & 1) { sum += ws[order[j]]; mask |= 1 << order[j]; }
		sums[m] = sum;
		cellMask[m] = mask;
	}
	const count = new Uint16Array(S);
	count[0] = 1; // the full face
	for (let m = 0; m < total - 1; m++) count[sums[m]]++;
	for (let sum = 0; sum < S; sum++) {
		if (!count[sum]) throw new Error("weights are not complete");
		off[sum + 1] = off[sum] + count[sum];
	}
	masks = new Uint16Array(total);
	const fill = off.slice(0, S);
	masks[fill[0]++] = total - 1;
	for (let m = 0; m < total - 1; m++) masks[fill[sums[m]]++] = cellMask[m];
	return { off, masks, legacy: false };
}
const options = (ord, sum) => ord.off[sum + 1] - ord.off[sum];
// Digit slots [sum, count, sum, count, ...], most significant first: non-power-of-2 counts,
// then powers of 2, each by count then sum (custom.html's buildRadix). Only the count sums
// 0, step, 2 step, ... a cycle shows get digits.
function buildRadix(ord, step = 1, count = ord.off.length - 1) {
	const keys = [];
	for (let sum = 0; sum < ord.off.length - 1; sum += step) {
		const m = options(ord, sum);
		if (m > 1 && sum / step < count) keys.push(((m & (m - 1)) === 0 ? 1 << 26 : 0) + m * 8192 + sum);
	}
	const sorted = Uint32Array.from(keys).sort(), slots = new Uint16Array(2 * sorted.length);
	sorted.forEach((k, i) => { slots[2 * i] = k % 8192; slots[2 * i + 1] = (k >> 13) % 8192; });
	return slots;
}
const engine = face => {
	const ordering = buildOrdering(face), S = face.W * face.H;
	return { face, S, ordering, radix: { [S]: buildRadix(ordering) } };
};
const defaultMod = S => S % 60 === 0 ? 60 : S; // a full face is one minute when it can be

// Display modes as on the web: [tick numerator, denominator (s), states per cycle]: seconds
// (mod per cycle), minutes in the hour, and hours as a 12-hour dial (lit area = elapsed share
// of 12 h) or, with the hours24 setting, the hour 0-23 as a sum of weights.
function modeParams(S, cfg, mode) {
	if (mode === 0) return [1, 1, cfg.mod || defaultMod(S)];
	if (mode === 1) return [60, 1, 60];
	return cfg.hours24 ? [3600, 1, 24] : [43200, S, S];
}

// Mirrors custom.html's faceState: cycles counted from origin (UTC) plus offset years. Like
// Wallpaper Engine, the minute and hour modes show local time (phone timezone incl. DST).
function clockState(eng, nowMs, cfg, mode = 0, local = null) {
	const { S, ordering } = eng;
	const [num, den, mod] = modeParams(S, cfg, mode);
	const step = S % mod === 0 ? S / mod : 1;
	// the default face keeps all 60 digits when it shows 0..mod-1 (its hours=24 panel)
	const key = mod === S || (ordering.legacy && step === 1) ? S : mod;
	const slots = eng.radix[key] ??= buildRadix(ordering, step, mod);
	const ms = divmod(mul(big(nowMs - cfg.origin), big(den)), num * 1000).q;
	const { q: cycle, r } = divmod(add(ms, mul(cfg.offset, big(31536000 * den / num))), mod);
	const ord = mode === 0 || !local ? r : mode === 1 ? local.getMinutes() : cfg.hours24 ? local.getHours()
		: Math.floor(((local.getHours() % 12) * 3600 + local.getMinutes() * 60 + local.getSeconds()) * S / 43200);
	return { slots, cycle, sum: ord * step, key: mode + ":" + str(cycle) };
}
// digit per sum for this cycle: floor divmod of cycle * P + N gives the digits of k mod period
function cycleDigits(eng, st, cfg) {
	let x = add(mul(st.cycle, cfg.P), cfg.N);
	const d = new Uint16Array(eng.S);
	for (let i = st.slots.length - 2; i >= 0; i -= 2) {
		const { q, r } = divmod(x, st.slots[i + 1]);
		d[st.slots[i]] = r;
		x = q;
	}
	return d;
}
const option = (eng, sum, i) => eng.ordering.masks[eng.ordering.off[sum] + i];
const shownMask = (eng, st, digits) => option(eng, st.sum, digits[st.sum]);

// decimal integer string mod m (m < 2^48)
function strMod(s, m) {
	const neg = s[0] === "-";
	let r = 0;
	for (let i = neg || s[0] === "+" ? 1 : 0; i < s.length; i++)
		r = (r * 10 + s.charCodeAt(i) - 48) % m;
	return neg && r ? m - r : r;
}

// "WxH;w.x.y.a.b,...;colours" as tiled by the phone (src/pkjs/face.js); "" is the default face
function parseFace(text) {
	if (!text) return LEGACY;
	const [grid, list, colors] = text.split(";");
	const g = /^(\d+)x(\d+)$/.exec(grid);
	if (!g || !list || !/^[0-3]*$/.test(colors || "")) throw new Error("bad face");
	const W = +g[1], H = +g[2], cells = list.split(",").map(c => c.split(".").map(Number));
	if (cells.length > 12 || W * H > 420 || colors.length !== cells.length) throw new Error("face too big");
	const seen = new Uint8Array(W * H);
	for (const c of cells) {
		const [w, x, y, a, b] = c;
		if (c.length !== 5 || c.some(v => !Number.isInteger(v) || v < 0) || a * b !== w || x + a > W || y + b > H)
			throw new Error("bad cell");
		for (let r = y; r < y + b; r++) for (let k = x; k < x + a; k++) if (seen[r * W + k]++) throw new Error("cells overlap");
	}
	if (seen.some(v => !v)) throw new Error("cells leave a gap");
	return { W, H, cells, colors: colors.split("").map(Number) };
}

const DEFAULTS = { origin: "1970-01-01T00:00:00Z", P: "1", N: "0", offset: "0", hours: "12", mono: "0", lit: "ff0000", unlit: "ffffff", face: "", mod: "" };
const P_MAX = 900719925474099; // keeps strMod(N, P) exact

// String settings -> config with the face engine; throws on invalid input.
function parseConfig(s) {
	const origin = Date.parse(s.origin);
	if (isNaN(origin)) throw new Error("bad origin");
	if (!/^\d+$/.test(s.P) || !/^\d+$/.test(s.N) || !/^[+-]?\d+$/.test(s.offset))
		throw new Error("P, N, offset must be integers (P, N >= 0)");
	if (s.hours !== "12" && s.hours !== "24") throw new Error("hours must be 12 or 24");
	const P = Number(s.P);
	if (P < 1 || P > P_MAX) throw new Error("P must be 1.." + P_MAX);
	const eng = engine(parseFace(s.face || ""));
	const mod = s.mod ? Number(s.mod) : 0;
	if (s.mod && !(/^\d+$/.test(s.mod) && mod >= 1 && eng.S % mod === 0)) throw new Error("mod must divide " + eng.S);
	const rgb = hex => {
		if (!/^[0-9a-fA-F]{6}$/.test(hex)) throw new Error("colours are rrggbb");
		return [0, 2, 4].map(i => parseInt(hex.slice(i, i + 2), 16));
	};
	return {
		origin, P: big(P), N: big(strMod(s.N, P)), offset: bigStr(s.offset), // hraj.si: N >= P -> N % P
		hours24: s.hours === "24", mono: s.mono === "1", // mono: every cell lit / unlit in one colour
		lit: rgb(s.lit || DEFAULTS.lit), unlit: rgb(s.unlit || DEFAULTS.unlit),
		mod, eng
	};
}
// --- end core ---

const KEYS = ["ORIGIN", "P", "N", "OFFSET", "HOURS", "MONO", "FACE", "MOD", "LIT", "UNLIT"];
const NAMES = { ORIGIN: "origin", P: "P", N: "N", OFFSET: "offset", HOURS: "hours", MONO: "mono", LIT: "lit", UNLIT: "unlit", FACE: "face", MOD: "mod" };

function loadSettings() {
	try {
		const s = { ...DEFAULTS, ...JSON.parse(localStorage.getItem("now") ?? "{}") };
		return parseConfig(s);
	} catch (e) {
		console.log("settings reset: " + e.message);
		return parseConfig(DEFAULTS);
	}
}
let cfg = loadSettings();

// --- drawing ---
const render = new Poco(screen);
const W = render.width, H = render.height;
const ROUND = W === H && W >= 260; // gabbro (Round 2): fit the grid inside the circle

const bg = render.makeColor(0, 0, 0); // blends with the grid frame and the bezel
const frame = render.makeColor(0, 0, 0);
const white = render.makeColor(255, 255, 255);
// custom.html's PALETTE: yellow, red, blue, dark (#303030 there, kept apart from the frame here)
const PALETTE = [render.makeColor(255, 255, 0), render.makeColor(255, 0, 0), render.makeColor(0, 0, 255), render.makeColor(85, 85, 85)];
const monoColors = () => [render.makeColor(...cfg.lit), render.makeColor(...cfg.unlit)];

// Square units, gap g between cells and around the frame, centred with black margins.
function layout(face) {
	const g = Math.max(2, Math.round(H / 76));
	let u = 40;
	for (; u > 1; u--) {
		const w = face.W * u + (face.W + 1) * g, h = face.H * u + (face.H + 1) * g;
		if (ROUND ? w * w + h * h <= W * W : w <= W && h <= H) break;
	}
	const GW = face.W * u + (face.W + 1) * g, GH = face.H * u + (face.H + 1) * g;
	const GX = (W - GW) >> 1, GY = (H - GH) >> 1;
	const rects = face.cells.map(([w, c, r, cs, rs], i) => ({
		color: PALETTE[face.colors[i]],
		x: GX + g + c * (u + g), y: GY + g + r * (u + g),
		w: cs * u + (cs - 1) * g, h: rs * u + (rs - 1) * g
	}));
	return { GX, GY, GW, GH, rects };
}
let geo = layout(cfg.eng.face), [monoLit, monoUnlit] = monoColors();

let digits = null, digitsK = "", lastMask = -1, full = true, mode = 0;

function draw(date) {
	const st = clockState(cfg.eng, date.getTime(), cfg, mode, date);
	if (st.key !== digitsK) {
		digits = cycleDigits(cfg.eng, st, cfg);
		digitsK = st.key;
	}
	const mask = shownMask(cfg.eng, st, digits);
	if (!full && mask === lastMask) return;
	const { GX, GY, GW, GH, rects } = geo;
	if (full) {
		render.begin();
		render.fillRectangle(bg, 0, 0, W, H);
		full = false;
	} else {
		render.begin(GX, GY, GW, GH);
	}
	render.fillRectangle(frame, GX, GY, GW, GH);
	for (let i = 0; i < rects.length; i++) {
		const r = rects[i];
		const on = mask & (1 << i);
		render.fillRectangle(cfg.mono ? (on ? monoLit : monoUnlit) : on ? r.color : white, r.x, r.y, r.w, r.h);
	}
	render.end();
	lastMask = mask;
}

// Seconds mode needs secondchange; the other modes only change once a minute (saves battery).
const onTime = e => draw(e.date);
let timeEvent = null;
function setMode(m) {
	mode = m;
	digitsK = "";
	lastMask = -1;
	const ev = mode === 0 ? "secondchange" : "minutechange";
	if (ev !== timeEvent) {
		if (timeEvent) watch.removeEventListener(timeEvent, onTime);
		timeEvent = ev;
		watch.addEventListener(ev, onTime); // also fires once right away
	} else {
		draw(new Date);
	}
}
setMode(0);
const nextMode = () => setMode((mode + 1) % 3);

// Flick the wrist to cycle the modes; PebbleOS doesn't deliver touch to watchfaces.
new Accelerometer({ onTap: nextMode });

// --- settings from the phone (config page in src/pkjs) ---
new Message({
	keys: KEYS,
	input: 512, // a few short strings and the face; the default takes 2 x 8 KB of app heap
	output: 64,
	onReadable() {
		const msg = this.read();
		const s = { ...DEFAULTS };
		msg.forEach((value, key) => {
			if (NAMES[key]) s[NAMES[key]] = String(value);
		});
		try {
			cfg = parseConfig(s);
			geo = layout(cfg.eng.face);
			[monoLit, monoUnlit] = monoColors();
			localStorage.setItem("now", JSON.stringify(s));
			digitsK = "";
			full = true;
			draw(new Date);
			console.log("settings applied: " + JSON.stringify(s));
		} catch (e) {
			console.log("settings rejected: " + e.message);
		}
	},
	onWritable() {},
	onSuspend() {}
});
