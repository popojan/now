// Phone side: settings page sent to the watch. The face is tiled and coloured here
// (face.js, as web/custom.html); the watch only gets the finished layout.
var faceText = require("./face").faceText;

var DEFAULTS = { origin: "1970-01-01T00:00:00Z", P: "1", N: "0", offset: "0", hours: "12", mono: "0",
	lit: "ff0000", unlit: "ffffff", bits: "", mod: "" };
// How the panel shows the 64 colours (index r*16 + g*4 + b, 2 bits each), from pebble-tool's
// screenshot colour correction (MIT)
var PANEL = "000000001e410043870068ca2b4a2c27514f16638d007dce5e98605c9b7257a5a24cb4db8ee3918ee69e8aebc084f5f14a161b48274840488a2f6bcc564e365454544f67904180d0759a64759d7671a6a469b5dd9ee5949de7a09becc295f6f299353f983e5a9556948f74d29d5b4d9d60649a70999587d5afa072aea382abababa7bae2c9e89dc9eaa7c7f0c8c3f9f7e35462e25874e16aa3de83dce66e6be6727ce37fa7e194dff1aa86f1ad93efb5b8ecc3ebffeeabfff1b5fff6d3ffffff";
var MONO_PRESETS = [["ff0000", "ffffff", "red on white (web mono=1)"], ["ffffff", "555555", "white on Mondrian black"]];
var PRESETS = [
	["", "Classic 6×10: 1, 2, 4, 6, 12, 15, 20 (60 s)"],
	["1,2,4,6,9,12,20,30,36", "10×12: 1, 2, 4, 6, 9, 12, 20, 30, 36 (120)"]
];

function load() {
	var s = {};
	try {
		s = JSON.parse(localStorage.getItem("now") || "{}") || {};
	} catch (e) {}
	var r = {};
	for (var k in DEFAULTS) r[k] = typeof s[k] === "string" ? s[k] : DEFAULTS[k];
	return r;
}

function esc(v) {
	return String(v).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
}

function option(value, label, current) {
	return '<option value="' + esc(value) + '"' + (value === current ? " selected" : "") + ">" + esc(label) + "</option>";
}

// Runs in the settings page: live S, divisors for mod, primes P should avoid, validation.
function pageScript() {
	function v(id) { return document.getElementById(id).value.trim(); }
	function weights() {
		var b = v("bits") || "1,2,4,6,12,15,20";
		return b.split(",").map(function (w) { return Number(w); });
	}
	// subsets summing to each s mod S (the full face counts at 0)
	function counts(ws, S) {
		var c = [1];
		for (var s = 1; s <= S; s++) c.push(0);
		ws.forEach(function (w) { for (var s = S; s >= w; s--) c[s] += c[s - w]; });
		c[0] += c[S];
		return c.slice(0, S);
	}
	function check() {
		var ws = weights(), S = 0, reach = 0, e = "";
		ws.forEach(function (w) { S += w; });
		if (ws.some(function (w) { return !(w >= 1 && w % 1 === 0); })) e = "weights must be positive integers";
		else if (ws.length > 12) e = "at most 12 cells";
		else if (S > 420) e = "the weights may sum to at most 420";
		else ws.slice().sort(function (a, b) { return a - b; }).forEach(function (w) {
			if (!e && w > reach + 1) e = "no subset of the weights sums to " + (reach + 1);
			reach += w;
		});
		var divs = [];
		if (!e) for (var d = 1; d <= S; d++) if (S % d === 0) divs.push(d);
		document.getElementById("S").textContent = e ? "" : "S = " + S + "; mod divides S: " + divs.join(", ") +
			" (empty: " + (S % 60 ? S : 60) + ")";
		var primes = {};
		if (!e) counts(ws, S).forEach(function (m) {
			for (var p = 2; m > 1; p++) while (m % p === 0) { primes[p] = 1; m /= p; }
		});
		document.getElementById("Phint").textContent = "signature period; coprime with " +
			(Object.keys(primes).join(", ") || "everything") + " keeps the full period";
		var mod = v("mod");
		if (!e && mod && !(/^\d+$/.test(mod) && S % Number(mod) === 0)) e = "mod must divide " + S;
		return e;
	}
	// 64-colour swatch grids for the mono colours, previewed as the panel shows them
	var panel = document.body.getAttribute("data-panel");
	function nominal(i) {
		return [i >> 4, i >> 2 & 3, i & 3].map(function (c) { return ("0" + (c * 85).toString(16)).slice(-2); }).join("");
	}
	function grid(id) {
		var box = document.getElementById(id + "-grid"), input = document.getElementById(id);
		box.innerHTML = "";
		for (var i = 0; i < 64; i++) (function (i) {
			var b = document.createElement("button");
			b.type = "button";
			b.className = "sw" + (nominal(i) === input.value ? " on" : "");
			b.style.background = "#" + panel.substr(6 * i, 6);
			b.title = "#" + nominal(i);
			b.onclick = function () { input.value = nominal(i); grid(id); };
			box.appendChild(b);
		})(i);
	}
	function grids() { grid("lit"); grid("unlit"); }
	[].forEach.call(document.querySelectorAll("[data-lit]"), function (b) {
		b.onclick = function () {
			document.getElementById("lit").value = b.getAttribute("data-lit");
			document.getElementById("unlit").value = b.getAttribute("data-unlit");
			document.getElementById("mono").value = "1";
			grids();
		};
	});
	grids();
	document.getElementById("preset").onchange = function () {
		if (this.value !== "-") document.getElementById("bits").value = this.value;
		check();
	};
	document.getElementById("bits").oninput = check;
	document.getElementById("mod").oninput = check;
	check();
	document.getElementById("save").onclick = function () {
		var s = { origin: v("origin"), offset: v("offset"), P: v("P"), N: v("N"), hours: v("hours"),
			mono: v("mono"), lit: v("lit"), unlit: v("unlit"), bits: v("bits").replace(/\s+/g, ""), mod: v("mod") }, e = check();
		if (e) {}
		else if (isNaN(Date.parse(s.origin))) e = "origin is not a valid date";
		else if (!/^-?\d+$/.test(s.offset)) e = "offset must be an integer";
		else if (!/^\d+$/.test(s.P) || s.P.replace(/^0+/, "") === "") e = "P must be a positive integer";
		else if (!/^\d+$/.test(s.N)) e = "N must be a non-negative integer";
		if (e) { document.getElementById("err").textContent = e; return; }
		// the emulator passes return_to; the phone app expects pebblejs://close#
		var r = new URLSearchParams(location.search).get("return_to") || "pebblejs://close#";
		location.href = r + encodeURIComponent(JSON.stringify(s));
	};
}

function page(s) {
	var known = PRESETS.some(function (p) { return p[0] === s.bits; });
	var sel = 'style="width:100%;font-size:16px;padding:8px"';
	return '<!doctype html><html><head><meta charset="utf-8">' +
		'<meta name="viewport" content="width=device-width,initial-scale=1">' +
		'<title>Now</title><style>' +
		'body{font-family:sans-serif;margin:16px;background:#e8e8e8;color:#1e1e1e}' +
		'label{display:block;margin:14px 0 4px;font-weight:bold}' +
		'input{width:100%;box-sizing:border-box;font-size:16px;padding:8px}' +
		'small{color:#555;display:block}#err{color:#c00;margin-top:12px}' +
		'button{margin-top:20px;width:100%;font-size:18px;padding:12px;border:0;background:#1e1e1e;color:#fff}' +
		'.grid{display:grid;grid-template-columns:repeat(16,1fr);gap:2px;margin-top:4px}' +
		'.grid button.sw{margin:0;padding:0;aspect-ratio:1;border:1px solid #888}' +
		'.grid button.sw.on{outline:3px solid #1e1e1e;outline-offset:1px;position:relative}' +
		'.presets button{margin-top:6px;font-size:15px;padding:8px;background:#555}' +
		'</style></head><body data-panel="' + PANEL + '"><h2>Now</h2>' +
		'<label>face</label><select id="preset" ' + sel + '>' +
		PRESETS.map(function (p) { return option(p[0], p[1], s.bits); }).join("") +
		option("-", "Own weights…", known ? "" : "-") + '</select>' +
		'<label>cell weights</label><input id="bits" value="' + esc(s.bits) + '" placeholder="1,2,4,6,12,15,20">' +
		'<small>complete: every sum 0..S is a subset; tiled to fit the display</small>' +
		'<label>mod (seconds per cycle)</label><input id="mod" inputmode="numeric" value="' + esc(s.mod) + '">' +
		'<small id="S"></small>' +
		'<label>origin</label><input id="origin" value="' + esc(s.origin) + '">' +
		'<small>ISO 8601, e.g. 2023-05-17T08:13:00Z</small>' +
		'<label>offset (years)</label><input id="offset" inputmode="numeric" value="' + esc(s.offset) + '">' +
		'<label>P</label><input id="P" inputmode="numeric" value="' + esc(s.P) + '">' +
		'<small id="Phint"></small>' +
		'<label>N</label><input id="N" inputmode="numeric" value="' + esc(s.N) + '">' +
		'<small>signature value, 0 &le; N &lt; P</small>' +
		'<label>hour panel</label><select id="hours" ' + sel + '>' +
		option("12", "12-hour dial: lit area = share of 12 h", s.hours) +
		option("24", "24 hours: hour as a sum of weights", s.hours) + '</select>' +
		'<label>colours</label><select id="mono" ' + sel + '>' +
		option("0", "Mondrian: yellow, red, blue, grey", s.mono) +
		option("1", "Mono: two colours below", s.mono) + '</select>' +
		'<div class="presets">' + MONO_PRESETS.map(function (p) {
			return '<button type="button" data-lit="' + p[0] + '" data-unlit="' + p[1] + '">' + esc(p[2]) + '</button>';
		}).join("") + '</div>' +
		'<label>mono: lit cells</label><input type="hidden" id="lit" value="' + esc(s.lit) + '"><div class="grid" id="lit-grid"></div>' +
		'<label>mono: unlit cells</label><input type="hidden" id="unlit" value="' + esc(s.unlit) + '"><div class="grid" id="unlit-grid"></div>' +
		'<small>as the panel shows them</small>' +
		'<div id="err"></div><button id="save">Save</button>' +
		'<script>(' + pageScript.toString() + ')();</script></body></html>';
}

// width / height of the display, for the grid nearest its shape
function displayShape() {
	try {
		var w = Pebble.getActiveWatchInfo && Pebble.getActiveWatchInfo();
		if (w && w.platform === "gabbro") return 1;
	} catch (e) {}
	return 200 / 228;
}

function send(s) {
	var face;
	try {
		face = faceText(s.bits, displayShape());
	} catch (e) {
		console.log("face not tiled: " + e.message);
		return;
	}
	Pebble.sendAppMessage({ ORIGIN: s.origin, P: s.P, N: s.N, OFFSET: s.offset, HOURS: s.hours || "12",
		MONO: s.mono || "0", LIT: s.lit || DEFAULTS.lit, UNLIT: s.unlit || DEFAULTS.unlit, FACE: face, MOD: s.mod || "" },
		function () { console.log("settings sent: face " + (face || "default")); },
		function (e) { console.log("settings send failed: " + JSON.stringify(e)); });
}

Pebble.addEventListener("ready", function () {
	console.log("Now: phone side ready");
});

Pebble.addEventListener("showConfiguration", function () {
	Pebble.openURL("data:text/html;charset=utf-8," + encodeURIComponent(page(load())));
});

Pebble.addEventListener("webviewclosed", function (e) {
	if (!e || !e.response) return;
	var s;
	try {
		s = JSON.parse(decodeURIComponent(e.response));
	} catch (err) {
		console.log("bad config response");
		return;
	}
	localStorage.setItem("now", JSON.stringify(s));
	send(s);
});
