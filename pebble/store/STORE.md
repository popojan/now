# Publishing Now to the Pebble appstore

Drafts and a checklist; nothing has been published. Based on the Pebble developer guides
([Publishing an App](https://developer.repebble.com/guides/appstore-publishing/publishing-an-app/),
[Preparing a Submission](https://developer.repebble.com/guides/appstore-publishing/preparing-a-submission/))
and `pebble publish` 5.2, as of 2026-11-20.

## Decide first

- [ ] **License.** The repository has no LICENSE file. The appstore does not ask for one, but
      the listing links to the source, and without a license the code is "all rights reserved".
- [ ] **Name.** The listing name defaults to `displayName` ("Now"). Something like
      "Now: Mondrian Clock" is easier to find; changing `displayName` changes it on the watch too.
- [ ] **Quick View.** The developer blog asks watchfaces to handle the timeline peek, which
      covers the bottom of the screen. Alloy fires `watch.addEventListener("resize", …)` when it
      opens or closes; the face would shrink or shift the grid then. Not handled yet.
- [ ] **Branch.** Merge `faces` into `main` and publish from `main`.
- [ ] **Version.** `package.json` says 1.0.0. Every release needs a new version (numbers and dots,
      each part at most 255). Keep the `uuid` forever: it ties releases to the listing.

## What the appstore takes

| Item | Now | Notes |
|---|---|---|
| .pbw | `build/now-pebble.pbw`, 53 KB | limit 4.4 MB; type watchface, platforms emery and gabbro come from the build |
| Name, description | drafts below | description is plain text; watchfaces get no category (listed under Faces) |
| Source URL | https://github.com/popojan/now | `pebble publish` offers the git remote |
| Website URL | optional, e.g. https://hraj.si/now | |
| Screenshots | `store/screenshots/` | up to 5 per platform, exact size, no watch frame: emery 200×228, gabbro 260×260 |
| Banner | optional, 720×320 | none yet |
| Icons | not used for watchfaces | generated from the name if missing |
| Release notes | per release, optional | draft below |

The screenshots were taken in the emulator with `pebble screenshot`, which maps colours to how
the panel shows them. File names start with the platform, as `pebble publish --screenshots`
requires. To retake one, set the face up in the emulator and run
`pebble screenshot --emulator emery --no-open store/screenshots/emery_1-classic-numbers.png`.

## Steps

1. `pebble login`: opens the browser for a Firebase (Pebble) account. The first
   `pebble publish` creates the linked developer account if there is none.
2. `pebble build`, then install on the watch and check the release build once more.
3. Publish a **draft** (a release is not installable until it is published):

   ```sh
   cd pebble
   pebble publish --non-interactive --no-gif-all-platforms \
     --description "$(sed -n '/^<!-- short -->/,/^<!-- end -->/p' store/STORE.md | sed '1d;$d')" \
     --source https://github.com/popojan/now \
     --screenshots store/screenshots/*.png \
     --release-notes "First release."
   ```

   Without `--is-published` the release stays a draft. Without `--no-gif-all-platforms` the
   tool also records rollover GIFs in the emulator.
4. In the Developer Dashboard (the only place where a listing can be edited): paste the long
   description, set visibility to **Unlisted**, publish the release, install it from the
   listing link (https://apps.repebble.com/<id>) and check it on the watch.
5. Switch visibility to **Listed**.
6. Updates: bump `version`, then `pebble publish --release-notes "…"` (add `--replace-screenshots`
   only to replace all screenshots; the CLI can't undo it).

The Rebble appstore (dev-portal.rebble.io) is separate and optional: it wants a screenshot for
every supported platform; a banner is optional for watchfaces.

## Listing text (drafts)

Short description:

<!-- short -->
A Mondrian clock where no two minutes are alike: the lit cells add up to the time.
<!-- end -->

Description:

> Now is a Mondrian-style clock where no two minutes are alike.
>
> Every second lights a set of coloured cells whose areas add up to the second. The classic face
> has seven cells worth 1, 2, 4, 6, 12, 15 and 20, sixty squares in all. Most seconds can be shown
> in several ways, and the watch chooses one by a signature of the current minute, so a minute's
> pattern does not come back for 2^30 · 3^16 minutes: about 88 billion years, more than six times
> the age of the universe.
>
> Learning to read it: the reading hints show what each cell is worth. Numbers on the lit cells
> are on from the start, and a faint unit grid can show every cell as its count of squares. Add up
> the lit numbers and you have the second. Turn the hints off once you can read it.
>
> Flick your wrist to switch between seconds, minutes (the lit share of the hour) and hours (the
> lit share of 12 or 24 hours, following your watch's 12/24h setting).
>
> Settings, in the Pebble app:
> • Face: the classic 6×10, a 10×12 face of 120 squares, or your own cell weights; the phone tiles
>   any complete set to fit your display.
> • Colours: Mondrian, or mono with your own lit, unlit and line colours. Presets include ink on
>   paper, silhouette and blueprint.
> • Hour panel, cycle length, and the clock's signature (origin, P, N) for the curious.
>
> The same clock runs in the browser at hraj.si/now and as a Wallpaper Engine wallpaper.
> Source code: github.com/popojan/now
>
> Privacy: Now collects no data. Settings stay on your phone and watch; the settings page runs
> locally and makes no network requests.

More faces to try under "Own weights…" (for the description or the GitHub README):

- 1, 2, 4, 8, 15, 30 on 6×10: almost binary and the easiest to learn, but the pattern repeats
  every 16 minutes.
- 1, 2, 4, 6, 9, 20, 36, 42 on 10×12: the squarest cells of the 120 faces.

Release notes, 1.0.0:

> First release: classic 6×10 and 120-square faces, any complete set of weights, reading hints,
> mono colours with presets, hour panel following the watch's 12/24h setting.
