// Checks the watch core (src/embeddedjs/main.js) against the Wallpaper Engine clock
// (wallpaper/now/now.html): signature P/N, origin, offset and all three display modes.
// Usage: TZ=Europe/Prague node test/core.test.js [dump.json]
const fs = require('fs');
const path = require('path');
const root = path.join(__dirname, '..', '..');
const src = fs.readFileSync(path.join(__dirname, '..', 'src', 'embeddedjs', 'main.js'), 'utf8');
const core = src.split('// --- core (shared with the node tests) ---')[1].split('// --- end core ---')[0];
const C = new Function(core + '; return { LEGACY, clockState, cycleDigits, shownMask, option, parseConfig, str };')();
const CELLS = C.LEGACY.cells.map(c => c[0]);

const we = fs.readFileSync(path.join(root, 'wallpaper', 'now', 'now.html'), 'utf8');
const ordering = eval('(' + we.match(/let ordering = (\{[\s\S]*?\});/)[1] + ')');
const perm = new Function('ordering',
	we.match(/function integerDigitsBasePad[\s\S]*?\n    }\n/)[0] +
	we.match(/function perm\(t\)[\s\S]*?return ret;\n    }/)[0] + '; return perm;')(ordering);
const CP = 46221064723759104n;
const HOUR_MODES = eval('(' + we.match(/const HOUR_MODES = (\{[^}]*\});/)[1] + ')');

// Wallpaper Engine's setInterval body plus the hraj.si offset term
// hours: the hour panel as Wallpaper Engine names it (12, day, 24)
function reference(now, origin, offset, P, N, mode, hours) {
	const [tick, period] = mode < 2 ? [[1n, 60n], [60n, 60n]][mode] : HOUR_MODES[hours];
	const h24 = mode === 2 && hours === "24";
	let t = BigInt(Math.floor((now - origin) / 1000.0));
	t = (t + period * tick * CP + offset * 365n * 24n * 3600n) % (60n * CP * period * tick);
	const k = (t / (period * tick) * P + N % P) % CP, d = new Date(now);
	const ord = mode === 0 ? (t / tick) % period : BigInt(mode === 1 ? d.getMinutes() : h24 ? d.getHours()
		: Math.floor(((hours === "day" ? d.getHours() : d.getHours() % 12) * 3600 + d.getMinutes() * 60 + d.getSeconds()) / Number(tick)));
	return ordering[ord][perm(k)[ord]];
}

let seed = 7;
const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
const big = d => BigInt(Math.floor(rnd() * 10 ** d));
const origins = ["1970-01-01T00:00:00Z", "2023-05-17T08:13:00Z", "2026-11-10T20:00:00+01:00", "-002000-01-01T00:00:00Z"];
const starts = ["2026-03-29T00:30:00Z", "2026-10-25T00:30:00Z", "2026-11-10T12:00:00Z"]; // EU DST switches
let n = 0, bad = 0;
const dump = [];
for (let i = 0; i < 2000; i++) {
	const s = {
		origin: origins[i % 4],
		P: String([1n, 7n, 25n, 1000003n, big(12) * 6n + 1n][i % 5]),
		N: String(big(14)),
		offset: String([0n, 5n, -3n, 88000000000n, big(15)][(i >> 2) % 5]),
		hours: ["auto", "12", "day", "24"][(i >> 1) % 4]
	};
	const hour12 = (i >> 3) % 2 === 0; // the watch's 12/24h setting, for hours=auto
	const panel = s.hours === "auto" ? (hour12 ? "12" : "day") : s.hours;
	const cfg = C.parseConfig(s);
	const base = i % 2 ? Date.parse(starts[i % 3]) + rnd() * 7.2e6 : Date.parse(starts[2]) + (rnd() - 0.5) * 4e11;
	for (let mode = 0; mode < 3; mode++) for (let d = 0; d < 3; d++) {
		const now = Math.floor(base + d * 37000);
		const st = C.clockState(cfg.eng, now, cfg, mode, new Date(now), hour12);
		const mask = C.shownMask(cfg.eng, st, C.cycleDigits(cfg.eng, st, cfg));
		const got = CELLS.filter((c, b) => mask & (1 << b));
		const exp = reference(now, cfg.origin, BigInt(s.offset), BigInt(s.P), BigInt(s.N), mode, panel);
		n++;
		if (JSON.stringify(got) !== JSON.stringify(exp) && bad++ < 5) console.log("MISMATCH", s, now, mode, got, exp);
	}
	if (i < 300) {
		const st = C.clockState(cfg.eng, base, cfg);
		const d = C.cycleDigits(cfg.eng, st, cfg);
		const k = ((BigInt(C.str(st.cycle)) * BigInt(s.P) + BigInt(s.N) % BigInt(s.P)) % CP + CP) % CP;
		dump.push([String(k), Array.from({ length: 60 }, (_, sum) => C.option(cfg.eng, sum, d[sum]))]);
	}
}
if (process.argv[2]) fs.writeFileSync(process.argv[2], JSON.stringify(dump));
console.log(`core vs wallpaper/now/now.html (TZ=${process.env.TZ || "system"}): ${n} checks, ${bad} mismatches`);
process.exit(bad ? 1 : 0);
