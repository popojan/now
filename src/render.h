/*
 * render.h - Visualization for "now" Mondrian clock
 *
 * Converts masks to text output. Supports various styles:
 * - ASCII or Unicode borders
 * - Multiple fill presets (blocks, CJK, emoji)
 * - Half-width and wide character modes
 */

#ifndef NOW_RENDER_H
#define NOW_RENDER_H

#include <stdint.h>
#include <stdio.h>

/* Grid layout: 6 columns x 10 rows
 * Each cell contains the cell value (1,2,4,6,12,15,20) that owns it */
extern const int GRID[10][6];

/* Render options */
typedef struct {
    int ascii;          /* 0=Unicode borders, 1=ASCII borders */
    int half_width;     /* 0=normal (2 cols/cell), 1=compact (1 col/cell) */
    int wide_fills;     /* 0=doubled fills, 1=wide chars (CJK/emoji) */
    const char *fills;  /* Custom 7 UTF-8 chars for cells 1,2,4,6,12,15,20, or NULL */
    const char *preset; /* Preset name, or NULL for default */
} render_opts_t;

/* Fill presets */
typedef struct {
    const char *name;
    const char *chars;
    int wide;
    int half;
} render_preset_t;

extern const render_preset_t PRESETS[];

/* Initialize render options with defaults */
void render_opts_init(render_opts_t *opts);

/* Apply a named preset to options */
int render_apply_preset(render_opts_t *opts, const char *preset_name);

/* Check if a cell value is visible in the mask */
int is_visible(uint8_t mask, int cell);

/* Render mask to FILE stream */
void render_mask(uint8_t mask, const render_opts_t *opts, FILE *out);

/* Render mask to a character grid (10 rows x 13 cols including borders)
 * Returns pointer to static buffer, not thread-safe */
const char *render_mask_to_str(uint8_t mask, const render_opts_t *opts);

/* Get list of preset names (NULL-terminated) */
const char **render_get_presets(void);

#endif /* NOW_RENDER_H */
