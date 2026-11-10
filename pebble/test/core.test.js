// Checks the watch core (src/embeddedjs/main.js) against the Wallpaper Engine clock
// (wallpaper/now/now.html): signature P/N, origin, offset and all three display modes.
// Usage: TZ=Europe/Prague node test/core.test.js [dump.json]
const fs = require('fs');
const path = require('path');
const root = path.join(__dirname, '..', '..');
const src = fs.readFileSync(path.join(__dirname, '..', 'src', 'embeddedjs', 'main.js'), 'utf8');
const core = src.split('// --- core (shared with the node test) ---')[1].split('// --- end core ---')[0];
const C = new Function(core + '; return { minuteMasks, clockState, parseConfig, CELLS };')();

const we = fs.readFileSync(path.join(root, 'wallpaper', 'now', 'now.html'), 'utf8');
const ordering = eval('(' + we.match(/let ordering = (\{[\s\S]*?\});/)[1] + ')');
const perm = new Function('ordering',
	we.match(/function integerDigitsBasePad[\s\S]*?\n    }\n/)[0] +
	we.match(/function perm\(t\)[\s\S]*?return ret;\n    }/)[0] + '; return perm;')(ordering);
const CP = 46221064723759104n;

// Wallpaper Engine's setInterval body plus the hraj.si offset term
function reference(now, origin, offset, P, N, mode) {
	const tick = [1n, 60n, 3600n][mode], period = [60n, 60n, 24n][mode];
	let t = BigInt(Math.floor((now - origin) / 1000.0));
	t = (t + period * tick * CP + offset * 365n * 24n * 3600n) % (60n * CP * period * tick);
	const k = (t / (period * tick) * P + N % P) % CP, d = new Date(now);
	const ord = mode === 0 ? (t / tick) % period : BigInt(mode === 1 ? d.getMinutes() : d.getHours());
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
		offset: String([0n, 5n, -3n, 88000000000n, big(15)][(i >> 2) % 5])
	};
	const cfg = C.parseConfig(s);
	const base = i % 2 ? Date.parse(starts[i % 3]) + rnd() * 7.2e6 : Date.parse(starts[2]) + (rnd() - 0.5) * 4e11;
	for (let mode = 0; mode < 3; mode++) for (let d = 0; d < 3; d++) {
		const now = Math.floor(base + d * 37000);
		const st = C.clockState(now, cfg, mode, new Date(now));
		const got = C.CELLS.filter((c, b) => C.minuteMasks(st.a, st.b)[st.second] & (1 << b));
		const exp = reference(now, cfg.origin, BigInt(s.offset), BigInt(s.P), BigInt(s.N), mode);
		n++;
		if (JSON.stringify(got) !== JSON.stringify(exp) && bad++ < 5) console.log("MISMATCH", s, now, mode, got, exp);
	}
	if (i < 300) {
		const st = C.clockState(base, cfg);
		dump.push([String(BigInt(st.a) * 1073741824n + BigInt(st.b)), C.minuteMasks(st.a, st.b)]);
	}
}
if (process.argv[2]) fs.writeFileSync(process.argv[2], JSON.stringify(dump));
console.log(`core vs wallpaper/now/now.html (TZ=${process.env.TZ || "system"}): ${n} checks, ${bad} mismatches`);
process.exit(bad ? 1 : 0);
