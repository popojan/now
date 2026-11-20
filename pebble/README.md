# Now for Pebble

The Now clock as a Pebble watchface for **Pebble Time 2** (emery) and **Pebble Round 2**
(gabbro), written in Alloy (JavaScript on Moddable XS). With the default face it computes the
same minutes as `wallpaper/now/now.html` and hraj.si/now, so `inverse/` can decode it. Other
faces follow `web/custom.html` (`bits=`, `mod=`).

- `k = (cycle * P + N) mod period`, with cycles counted from `origin` (UTC) plus `offset` years;
  on the default face a cycle is a minute and the period 2^30 * 3^16
- the `face` setting takes any complete set of cell weights, e.g. the preset
  1, 2, 4, 6, 9, 12, 20, 30, 36 (S = 120, 10 × 12). The phone tiles and colours it for the
  display (`src/pkjs/face.js`, copied from `custom.html`); the watch draws square units with
  black margins
- `mod`, a divisor of S: seconds per cycle. By default a full face is one minute when S is a
  multiple of 60 (S = 120: second s lights 2s), otherwise S seconds. `mod=120` on the
  120 face uses every sum, in a 2-minute cycle
- flick the wrist to cycle seconds → minutes in the hour → hours (PebbleOS gives watchfaces
  no touch or buttons);
  the last two show local time (phone timezone, DST included), like the Wallpaper Engine version
- the hour panel is a 12-hour dial: the lit area is the elapsed share of 12 hours, in steps of
  12 h / S (10:00 lights 50/60); the `hours` setting 24 brings back hour 0–23 as a sum of weights
- the `mono` setting lights every cell red, like `mono=1` on the web
- `face`, `mod`, `origin`, `offset`, `P`, `N`, `hours` and `mono` are set from the phone, see [Settings](#settings)

## Settings

Open the watchface's settings in the Pebble phone app. The form (`src/pkjs/index.js`) takes the
cell weights (a preset or your own, at most 12 cells summing to at most 420) and `mod`, and lists
the divisors of S and the primes P should avoid. It sends the tiled face, `mod`, `origin`
(ISO 8601), `offset` (years), `P` and `N` (as on hraj.si/now: N ≥ P is taken mod P), the hour
panel (`12` or `24`) and the colours (Mondrian or mono red) to the watch, which keeps them
across restarts. In the emulator, `pebble emu-app-config
--emulator emery` opens the same form in the browser.

## Build and run

```sh
pebble build
pebble install --emulator emery     # or gabbro
pebble emu-tap --emulator emery     # switch mode
pebble install --phone <ip>         # Developer Connection in the Pebble app
```

On the phone, tick Settings → Use LAN developer connection, then open Devices, pick the
watch and tap Dev Connection; only then does the app listen. Keep it in the foreground (iOS).
On Android, `pebble install --adb` does this over USB.

## Notes

- XS on Pebble ships without BigInt (`Error: dead strip`). The core uses its own signed big
  integers (base-10^7 limbs); dividing cycle * P + N by each digit's radix in turn gives the
  digits of k mod period directly. P is limited to ~9·10^14, `offset` is not.
- The XS heap holds only about 1500 slots: the face tables are typed arrays.
- pkjs is bundled by webpack 1 (acorn 3): ES6 at most, no `**`, `??` or object spread.
- The default XS heaps are too small (`memory full`); `src/c/mdbl.c` passes a
  `ModdableCreationRecord`. Its stack, slot and chunk must all be non-zero
  ([pebbleos#1621](https://github.com/coredevices/pebbleos/issues/1621)).
- After a crash the emulator can keep showing "Alloy: Fatal Error"; `pebble kill` and reinstall.

## Tests

```sh
test/run.sh
```

Compares the core with `wallpaper/now/now.html` (all modes, several time zones and DST
switches) and with `inverse/clock_inverse.py` on the default face, and with `web/custom.html`
on others (every mode and divisor `mod`); checks that `src/pkjs/face.js` is still a verbatim
copy of `custom.html`'s tiling and colouring.
