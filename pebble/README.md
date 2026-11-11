# Now for Pebble

The Now clock as a Pebble watchface for **Pebble Time 2** (emery) and **Pebble Round 2**
(gabbro), written in Alloy (JavaScript on Moddable XS). It computes the same minutes as
`wallpaper/now/now.html` and hraj.si/now, so `inverse/` can decode it.

- `k = (minute * P + N) mod 2^30 * 3^16`, with minutes counted from `origin` (UTC) plus `offset` years
- tap the screen or flick the wrist to cycle seconds → minutes in the hour → hours;
  the last two show local time (phone timezone, DST included), like the Wallpaper Engine version
- the hour panel is a 12-hour dial: the lit area is the elapsed share of 12 hours, in 12-minute
  steps (10:00 lights 50/60); the `hours` setting 24 brings back hour 0–23 as a sum of weights
- `origin`, `offset`, `P`, `N` and `hours` are set from the phone, see [Settings](#settings)

## Settings

Open the watchface's settings in the Pebble phone app. The form (`src/pkjs/index.js`) sends
`origin` (ISO 8601), `offset` (years), `P` and `N` (as on hraj.si/now: N ≥ P is taken mod P)
and the hour panel (`12` or `24`) to the watch, which keeps them across restarts. In the emulator, `pebble emu-app-config
--emulator emery` opens the same form in the browser.

## Build and run

```sh
pebble build
pebble install --emulator emery     # or gabbro
pebble emu-tap --emulator emery     # switch mode
pebble install --phone <ip>         # Developer Connection in the Pebble app
```

## Notes

- XS on Pebble ships without BigInt (`Error: dead strip`). The core works mod 2^30 and
  mod 3^16 separately and recombines them (CRT), so P is limited to ~9·10^14.
- The default XS heaps are too small (`memory full`); `src/c/mdbl.c` passes a
  `ModdableCreationRecord`. Its stack, slot and chunk must all be non-zero
  ([pebbleos#1621](https://github.com/coredevices/pebbleos/issues/1621)).
- After a crash the emulator can keep showing "Alloy: Fatal Error"; `pebble kill` and reinstall.

## Tests

```sh
test/run.sh
```

Compares the core with `wallpaper/now/now.html` (all modes, several time zones and DST
switches) and with `inverse/clock_inverse.py`.
