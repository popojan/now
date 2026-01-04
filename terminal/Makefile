# Mondrian Terminal Clock
# Compile: make
# Windows: cl now.c /Fe:now.exe (Visual Studio) or gcc on MinGW

CC = gcc
CFLAGS = -Os -Wall -s

now: now.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f now now.exe

test: now
	./test_inverse.sh

.PHONY: clean test
