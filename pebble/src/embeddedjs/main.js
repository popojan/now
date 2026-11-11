import Poco from "commodetto/Poco";
import Message from "pebble/message";
import Accelerometer from "embedded:sensor/Accelerometer";

// --- core (shared with the node test) ---
// No BigInt: XS on Pebble strips it ("dead strip"). The period 2^30 * 3^16 has coprime factors,
// so all arithmetic runs mod 2^30 and mod 3^16 separately and is recombined (CRT) per minute.
const B = 1073741824;  // 2^30
const A = 43046721;    // 3^16
const CELLS = [1, 2, 4, 6, 12, 15, 20];
// ordering from hraj.si/now as bitmasks (bit i = CELLS[i]); ORDERING[s] lists combos summing to s
const ORDERING = [
	[127,0],[1],[2],[3],[4],[5],[8,6],[9,7],[10],[11],
	[12],[13],[16,14],[17,15],[18],[32,19],[33,20],[34,21],[35,24,22],[36,25,23],
	[64,37,26],[65,40,38,27],[66,41,39,28],[67,42,29],[68,43,30],[69,44,31],[72,70,45],[73,71,48,46],[49,74,47],[75,50],
	[51,76],[77,52],[80,53,78],[81,79,56,54],[82,57,55],[96,83,58],[97,84,59],[98,85,60],[99,88,86,61],[100,89,87,62],
	[101,90,63],[104,102,91],[105,103,92],[106,93],[107,94],[108,95],[109],[112,110],[113,111],[114],
	[115],[116],[117],[120,118],[121,119],[122],[123],[124],[125],[126]
];

// Same digit order as perm(): 18 bits, then 16 trits, then 6 base-4 digits from the low 12 bits.
const SPEC = (() => {
	let n2 = 0, n3 = 0, n4 = 0;
	return ORDERING.map(c => {
		switch (c.length) {
			case 2: return [2, 2 ** (29 - n2++)];
			case 3: return [3, 3 ** (15 - n3++)];
			case 4: return [4, 4 ** (5 - n4++)];
			default: return [1, 1];
		}
	});
})();

const mod = (x, m) => ((x % m) + m) % m;
// x * y mod m for 0 <= x, y < m < 2^31, exact in doubles
function mulmod(x, y, m) {
	return (((x * Math.floor(y / 65536)) % m) * 65536 + x * (y % 65536)) % m;
}
// decimal integer string mod m (m < 2^48)
function strMod(s, m) {
	const neg = s[0] === "-";
	let r = 0;
	for (let i = neg || s[0] === "+" ? 1 : 0; i < s.length; i++)
		r = (r * 10 + s.charCodeAt(i) - 48) % m;
	return neg && r ? m - r : r;
}
// 2^30 inverse mod 3^16
const INV_B = (() => {
	let r0 = A, r1 = B % A, s0 = 0, s1 = 1;
	while (r1) {
		const q = Math.floor(r0 / r1);
		[r0, r1] = [r1, r0 - q * r1];
		[s0, s1] = [s1, s0 - q * s1];
	}
	return mod(s0, A);
})();

// Masks for all 60 seconds of minute k = a * 2^30 + b (a < 3^16, b < 2^30)
function minuteMasks(a, b) {
	return ORDERING.map((combos, s) => {
		const [base, div] = SPEC[s];
		return combos[Math.floor((base === 3 ? a : b) / div) % base];
	});
}

// Display modes as in hraj.si / Wallpaper Engine: [tick (s), period]: seconds, minutes in
// the hour, and hours as a 12-hour dial (lit area = elapsed share of 12 h, 12-minute steps)
// or, with the hours24 setting, the original hour 0-23 as a sum of weights.
const MODES = [[1, 60], [60, 60], [720, 60]];
const HOURS24 = [3600, 24];

// Mirrors hraj.si/now: k = (minute * P + N) mod 2^30*3^16, minute counted from origin (UTC).
// Like Wallpaper Engine, the minute and hour modes show local time (phone timezone incl. DST).
function clockState(nowMs, cfg, mode = 0, local = null) {
	const h24 = mode === 2 && cfg.hours24;
	const [tick, period] = h24 ? HOURS24 : MODES[mode], q = tick * period;
	const secs = Math.floor((nowMs - cfg.origin) / 1000);
	const units = Math.floor(secs / q), perYear = 31536000 / q;
	const mA = mod(units + mulmod(cfg.OA, perYear, A), A);
	const mB = mod(units + mulmod(cfg.OB, perYear, B), B);
	const kA = mod(mulmod(mA, cfg.PA, A) + cfg.NA, A);
	const kB = mod(mulmod(mB, cfg.PB, B) + cfg.NB, B);
	const ord = mode === 0 || !local ? mod(Math.floor(secs / tick), period)
		: mode === 1 ? local.getMinutes() : h24 ? local.getHours()
		: Math.floor(((local.getHours() % 12) * 3600 + local.getMinutes() * 60 + local.getSeconds()) / 720);
	return { a: mulmod(mod(kA - kB, A), INV_B, A), b: kB, second: ord };
}

const DEFAULTS = { origin: "1970-01-01T00:00:00Z", P: "1", N: "0", offset: "0", hours: "12" };
const P_MAX = 900719925474099; // keeps strMod(N, P) exact

// String settings -> residues; throws on invalid input.
function parseConfig(s) {
	const origin = Date.parse(s.origin);
	if (isNaN(origin)) throw new Error("bad origin");
	if (!/^\d+$/.test(s.P) || !/^\d+$/.test(s.N) || !/^[+-]?\d+$/.test(s.offset))
		throw new Error("P, N, offset must be integers (P, N >= 0)");
	if (s.hours !== "12" && s.hours !== "24") throw new Error("hours must be 12 or 24");
	const P = Number(s.P);
	if (P < 1 || P > P_MAX) throw new Error("P must be 1.." + P_MAX);
	const N = strMod(s.N, P); // hraj.si: N >= P -> N % P
	return {
		origin, PA: P % A, PB: P % B, NA: N % A, NB: N % B,
		OA: strMod(s.offset, A), OB: strMod(s.offset, B), hours24: s.hours === "24"
	};
}
// --- end core ---

const KEYS = ["ORIGIN", "P", "N", "OFFSET", "HOURS"];

function loadSettings() {
	try {
		const s = { ...DEFAULTS, ...JSON.parse(localStorage.getItem("now") ?? "{}") };
		return { settings: s, cfg: parseConfig(s) };
	} catch (e) {
		console.log("settings reset: " + e.message);
		return { settings: { ...DEFAULTS }, cfg: parseConfig(DEFAULTS) };
	}
}
let { cfg } = loadSettings();

// --- drawing ---
const render = new Poco(screen);
const W = render.width, H = render.height;
const ROUND = W === H && W >= 260; // gabbro (Round 2): fit the grid inside the circle

// Grid: 6x10 units, gap g between cells and around the frame (like the web .clock grid).
const g = Math.max(2, Math.round(H / 76));
let u = 40;
for (;;) {
	const w = 6 * u + 7 * g, h = 10 * u + 11 * g;
	if (ROUND ? w * w + h * h <= W * W : w <= W && h <= H) break;
	u--;
}
const GW = 6 * u + 7 * g, GH = 10 * u + 11 * g;
const GX = (W - GW) >> 1, GY = (H - GH) >> 1;

const bg = render.makeColor(170, 170, 170);
const frame = render.makeColor(0, 0, 0);
const white = render.makeColor(255, 255, 255);
const yellow = render.makeColor(255, 255, 0);
const red = render.makeColor(255, 0, 0);
const blue = render.makeColor(0, 0, 255);
const dark = render.makeColor(85, 85, 85); // web uses #303030 on a #1E1E1E frame; keep them distinguishable

// [cell value, color, col, row, colSpan, rowSpan] in CELLS order
const RECTS = [
	[1, dark, 3, 5, 1, 1],
	[2, yellow, 3, 6, 1, 2],
	[4, dark, 4, 6, 2, 2],
	[6, blue, 3, 8, 3, 2],
	[12, blue, 4, 0, 2, 6],
	[15, red, 0, 5, 3, 5],
	[20, yellow, 0, 0, 4, 5]
].map(([v, color, c, r, cs, rs]) => ({
	color,
	x: GX + g + c * (u + g), y: GY + g + r * (u + g),
	w: cs * u + (cs - 1) * g, h: rs * u + (rs - 1) * g
}));

let masks = null, maskK = "", lastMask = -1, full = true, mode = 0;

function draw(date) {
	const st = clockState(date.getTime(), cfg, mode, date);
	const key = st.a + ":" + st.b;
	if (key !== maskK) {
		masks = minuteMasks(st.a, st.b);
		maskK = key;
	}
	const mask = masks[st.second];
	if (!full && mask === lastMask) return;
	if (full) {
		render.begin();
		render.fillRectangle(bg, 0, 0, W, H);
		full = false;
	} else {
		render.begin(GX, GY, GW, GH);
	}
	render.fillRectangle(frame, GX, GY, GW, GH);
	for (let i = 0; i < 7; i++) {
		const r = RECTS[i];
		render.fillRectangle(mask & (1 << i) ? r.color : white, r.x, r.y, r.w, r.h);
	}
	render.end();
	lastMask = mask;
}

// Seconds mode needs secondchange; the other modes only change once a minute (saves battery).
const onTime = e => draw(e.date);
let timeEvent = null;
function setMode(m) {
	mode = m;
	maskK = "";
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
const nextMode = () => setMode((mode + 1) % MODES.length);

// Tap the screen (Time 2 / Round 2) or flick the wrist to cycle the modes.
try {
	let down = false;
	new device.sensor.Touch({
		onSample() {
			const p = this.sample();
			if (!p) return;
			if (p.length && !down) nextMode();
			down = p.length > 0;
		}
	});
} catch (e) {
	console.log("no touch: " + e.message);
}
new Accelerometer({ onTap: nextMode });

// --- settings from the phone (config page in src/pkjs) ---
new Message({
	keys: KEYS,
	input: 256, // settings are a few short strings; the default takes 2 x 8 KB of app heap
	output: 64,
	onReadable() {
		const msg = this.read();
		const s = { ...DEFAULTS };
		msg.forEach((value, key) => {
			const name = { ORIGIN: "origin", P: "P", N: "N", OFFSET: "offset", HOURS: "hours" }[key];
			if (name) s[name] = String(value);
		});
		try {
			cfg = parseConfig(s);
			localStorage.setItem("now", JSON.stringify(s));
			maskK = "";
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
