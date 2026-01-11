#!/bin/bash
# Video integration tests - validates k, origin, and signature against ground truth
cd "$(dirname "$0")"

PASS=0
FAIL=0

test_video() {
    local video="$1"
    local expected_k="$2"
    local expected_origin="$3"
    local expected_P="$4"
    local expected_N="$5"

    output=$(./venv/bin/python main.py "../test_videos/$video" 2>&1)
    detected_k=$(echo "$output" | grep "Minute identifier" | grep -oE '[0-9]+')
    detected_origin=$(echo "$output" | grep -A1 "Clock origin" | tail -1 | tr -d ' ')
    detected_P=$(echo "$output" | grep "P:" | grep -oE '[0-9]+')
    detected_N=$(echo "$output" | grep "N:" | grep -oE '[0-9]+')
    matches=$(echo "$output" | grep "matched" | grep -oE '[0-9]+/[0-9]+')

    if [ "$detected_k" = "$expected_k" ] && [ "$detected_origin" = "$expected_origin" ] && \
       [ "$detected_P" = "$expected_P" ] && [ "$detected_N" = "$expected_N" ]; then
        echo "PASS $video: k=$detected_k origin=$detected_origin P=$detected_P N=$detected_N ($matches)"
        PASS=$((PASS + 1))
    else
        echo "FAIL $video:"
        echo "  Expected: k=$expected_k origin=$expected_origin P=$expected_P N=$expected_N"
        echo "  Got:      k=$detected_k origin=$detected_origin P=$detected_P N=$detected_N ($matches)"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== Video Integration Tests ==="
echo

# Videos recorded before UTC change (local time origin = 1969-12-31 23:00:00 UTC)
# Standard encoding: P=1, N=0
test_video "IMG_6703.MOV" "29457038" "1969-12-3123:00:00" "1" "0"
test_video "IMG_6707.MOV" "29457282" "1969-12-3123:00:00" "1" "0"
test_video "IMG_6709.MOV" "29457321" "1969-12-3123:00:00" "1" "0"

# Videos recorded after UTC change (UTC origin = 1970-01-01 00:00:00 UTC)
# Standard encoding: P=1, N=0
test_video "IMG_6712.MOV" "29457295" "1970-01-0100:00:00" "1" "0"
test_video "IMG_6717.MOV" "13679798" "2000-01-0100:00:00" "1" "0"
test_video "IMG_6719.MOV" "13679847" "2000-01-0100:00:00" "1" "0"
test_video "IMG_6743.MOV" "29458875" "1970-01-0100:00:00" "1" "0"

# Ancient date test (Year 0 = 1 B.C. in ISO 8601)
# Standard encoding: P=1, N=0
test_video "IMG_6746.MOV" "1065579753" "0000-01-0100:00:00" "1" "0"

# Signature encoding tests
# Raw k = minute * P + N, decoded minutes = (k - N) / P
test_video "IMG_6782.MOV" "241355295366" "1970-01-0100:00:00" "8191" "1983"

# Monochrome red with signature (tests time-flow based empty color detection)
test_video "IMG_6789.MOV" "36381401836815" "1970-01-0100:00:00" "1234567" "196"

# Wallpaper Engine recording with signature
test_video "IMG_6791.MOV" "248953580" "1983-01-0100:00:00" "11" "7"

echo
echo "=== Results ==="
echo "Passed: $PASS"
echo "Failed: $FAIL"

if [ $FAIL -gt 0 ]; then
    exit 1
fi
echo "All video tests passed!"
