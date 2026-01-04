#!/bin/bash
# Unit tests for C inverse algorithm

set -e
cd "$(dirname "$0")"

# Ensure binary exists (normally built via 'make test')
if [ ! -f now ]; then
    make now
fi

PASS=0
FAIL=0

test_roundtrip() {
    local desc="$1"
    local args="$2"
    local expected="$3"

    result=$(./now $args -s -n 60 | ./now -i 2>/dev/null | tail -1)
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

# Basic test - default origin (epoch) with live time calculation
# This verifies the full round-trip works: generate -> parse -> recover origin
test_roundtrip "Default origin (epoch)" "" "1970-01-01T00:00:00Z"

# ASCII mode tests (same origin, different rendering)
test_roundtrip "ASCII mode" "-a" "1970-01-01T00:00:00Z"
test_roundtrip "ASCII distinct" "-a -d" "1970-01-01T00:00:00Z"

# Custom origin tests
test_roundtrip "Y2K origin" "-o 2000-01-01T00:00:00Z" "2000-01-01T00:00:00Z"
test_roundtrip "Custom origin 2020" "-o 2020-06-15T12:30:00Z" "2020-06-15T12:30:00Z"

echo
echo "=== Results ==="
echo "Passed: $PASS"
echo "Failed: $FAIL"

if [ $FAIL -gt 0 ]; then
    exit 1
fi
echo "All tests passed!"
