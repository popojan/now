#!/bin/bash
# Unit tests for C inverse algorithm

set -e
cd "$(dirname "$0")/.."

# Ensure binary exists
if [ ! -f bin/now ]; then
    make
fi

NOW=bin/now
PASS=0
FAIL=0

test_roundtrip() {
    local desc="$1"
    local args="$2"
    local expected="$3"

    # Note: -s flag needed for both encoder (instant generation) and decoder (instant input)
    # Use 180 frames (3 minutes) to ensure enough data for signature auto-detection
    result=$($NOW $args -s -n 180 | $NOW -i -s 2>/dev/null | grep "^origin:" | cut -d' ' -f2)
    if [ "$result" = "$expected" ]; then
        echo "✓ $desc"
        PASS=$((PASS + 1))
    else
        echo "✗ $desc: expected '$expected', got '$result'"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== C Inverse Round-trip Tests ==="
echo

# Basic test - default origin (epoch)
test_roundtrip "Default origin (epoch)" "" "1970-01-01T00:00:00Z"

# ASCII mode tests (same origin, different rendering)
test_roundtrip "ASCII mode" "-a" "1970-01-01T00:00:00Z"

# Custom origin tests
test_roundtrip "Y2K origin" "-o 2000-01-01T00:00:00Z" "2000-01-01T00:00:00Z"
test_roundtrip "Custom origin 2020" "-o 2020-06-15T12:30:00Z" "2020-06-15T12:30:00Z"

# Signature tests
test_roundtrip "Signature P=7" "-P 7" "1970-01-01T00:00:00Z"
test_roundtrip "Signature P=7 N=3" "-P 7 -N 3" "1970-01-01T00:00:00Z"

echo
echo "=== Results ==="
echo "Passed: $PASS"
echo "Failed: $FAIL"

if [ $FAIL -gt 0 ]; then
    exit 1
fi
echo "All tests passed!"
