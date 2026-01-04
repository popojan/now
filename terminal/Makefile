# Mondrian Terminal Clock
# Compile: make
# Windows: cl clock.c /Fe:clock.exe (Visual Studio) or gcc on MinGW

CC = gcc
CFLAGS = -Os -Wall -s

clock: clock.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f clock clock.exe

.PHONY: clean
