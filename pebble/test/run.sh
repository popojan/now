#!/bin/sh
# Cross-check the watch core against the Wallpaper Engine clock, web/custom.html and inverse/.
set -e
cd "$(dirname "$0")"
dump=$(mktemp)
trap 'rm -f "$dump"' EXIT
TZ=Europe/Prague node core.test.js "$dump"
TZ=Asia/Kolkata node core.test.js
TZ=UTC node core.test.js
TZ=Europe/Prague node face.test.js
TZ=Asia/Kolkata node face.test.js
python3 inverse_check.py "$dump"
