// Phone side: settings page (origin, offset, signature P and N) sent to the watch.
var DEFAULTS = { origin: "1970-01-01T00:00:00Z", P: "1", N: "0", offset: "0" };

function load() {
	try {
		var s = JSON.parse(localStorage.getItem("now") || "{}");
		return {
			origin: s.origin || DEFAULTS.origin, P: s.P || DEFAULTS.P,
			N: s.N || DEFAULTS.N, offset: s.offset || DEFAULTS.offset
		};
	} catch (e) {
		return DEFAULTS;
	}
}

function esc(v) {
	return String(v).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
}

function page(s) {
	return '<!doctype html><html><head><meta charset="utf-8">' +
		'<meta name="viewport" content="width=device-width,initial-scale=1">' +
		'<title>Now</title><style>' +
		'body{font-family:sans-serif;margin:16px;background:#e8e8e8;color:#1e1e1e}' +
		'label{display:block;margin:14px 0 4px;font-weight:bold}' +
		'input{width:100%;box-sizing:border-box;font-size:16px;padding:8px}' +
		'small{color:#555}#err{color:#c00;margin-top:12px}' +
		'button{margin-top:20px;width:100%;font-size:18px;padding:12px;border:0;background:#1e1e1e;color:#fff}' +
		'</style></head><body><h2>Now</h2>' +
		'<label>origin</label><input id="origin" value="' + esc(s.origin) + '">' +
		'<small>ISO 8601, e.g. 2023-05-17T08:13:00Z</small>' +
		'<label>offset (years)</label><input id="offset" inputmode="numeric" value="' + esc(s.offset) + '">' +
		'<label>P</label><input id="P" inputmode="numeric" value="' + esc(s.P) + '">' +
		'<small>signature period, coprime with 2 and 3 keeps the full period</small>' +
		'<label>N</label><input id="N" inputmode="numeric" value="' + esc(s.N) + '">' +
		'<small>signature value, 0 &le; N &lt; P</small>' +
		'<div id="err"></div><button id="save">Save</button>' +
		'<script>' +
		'function v(id){return document.getElementById(id).value.trim();}' +
		'document.getElementById("save").onclick=function(){' +
		'var s={origin:v("origin"),offset:v("offset"),P:v("P"),N:v("N")},e="";' +
		'if(isNaN(Date.parse(s.origin)))e="origin is not a valid date";' +
		'else if(!/^-?\\d+$/.test(s.offset))e="offset must be an integer";' +
		'else if(!/^\\d+$/.test(s.P)||s.P.replace(/^0+/,"")==="")e="P must be a positive integer";' +
		'else if(!/^\\d+$/.test(s.N))e="N must be a non-negative integer";' +
		'if(e){document.getElementById("err").textContent=e;return;}' +
		// the emulator passes return_to; the phone app expects pebblejs://close#
		'var r=new URLSearchParams(location.search).get("return_to")||"pebblejs://close#";' +
		'location.href=r+encodeURIComponent(JSON.stringify(s));};' +
		'</script></body></html>';
}

function send(s) {
	Pebble.sendAppMessage({ ORIGIN: s.origin, P: s.P, N: s.N, OFFSET: s.offset },
		function () { console.log("settings sent"); },
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
