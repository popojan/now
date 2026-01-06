# Makefile for "now" Mondrian clock

CC = gcc
CFLAGS = -O2 -Wall -Wextra
SRCDIR = src
BINDIR = bin

SRCS = $(SRCDIR)/core.c $(SRCDIR)/render.c $(SRCDIR)/now.c
HDRS = $(SRCDIR)/core.h $(SRCDIR)/render.h
TARGET = $(BINDIR)/now

.PHONY: all clean test test-roundtrip test-signatures

all: $(TARGET)

$(BINDIR):
	mkdir -p $(BINDIR)

$(TARGET): $(SRCS) $(HDRS) | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $(SRCS)

clean:
	rm -rf $(BINDIR)

# Test: round-trip (generate -> inverse -> verify)
test-roundtrip: $(TARGET)
	@echo "=== Round-trip test (P auto-detect) ==="
	$(TARGET) -P 7 -n 180 -s | $(TARGET) -i
	@echo ""
	@echo "=== Round-trip test (P specified) ==="
	$(TARGET) -P 60 -n 120 -s | $(TARGET) -P 60 -i

# Test: signatures display
test-signatures: $(TARGET)
	@echo "=== Signatures for P=60 (divisors: 1,2,3,4,5,6,10,12,15,20,30,60) ==="
	$(TARGET) -P 60 --sig -n 5 -s

# Test: different presets
test-presets: $(TARGET)
	@echo "=== Preset: blocks ===" && $(TARGET) -p blocks -n 1 -s
	@echo "=== Preset: cjk ===" && $(TARGET) -p cjk -n 1 -s
	@echo "=== Preset: emoji ===" && $(TARGET) -p emoji -n 1 -s

test: test-roundtrip test-signatures

# Install to /usr/local/bin
install: $(TARGET)
	install -m 755 $(TARGET) /usr/local/bin/now
