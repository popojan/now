/*
 * render.c - Visualization implementation for "now" clock
 */

#include "render.h"
#include <string.h>

/* Grid layout: which cell owns each position */
const int GRID[10][6] = {
    {20,20,20,20,12,12}, {20,20,20,20,12,12}, {20,20,20,20,12,12},
    {20,20,20,20,12,12}, {20,20,20,20,12,12}, {15,15,15,1,12,12},
    {15,15,15,2,4,4}, {15,15,15,2,4,4}, {15,15,15,6,6,6}, {15,15,15,6,6,6}
};

/* Fill presets: name, chars, wide, half */
const render_preset_t PRESETS[] = {
    {"blocks",   "\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88", 0, 0},  /* ███████ */
    {"blocks1",  "\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88", 0, 1},  /* Half-width */
    {"distinct", "\xe2\x96\x92\xe2\x96\x93\xe2\x96\x88\xe2\x96\x91\xe2\x96\x91\xe2\x96\x88\xe2\x96\x93", 0, 0},  /* ▒▓█░░█▓ */
    {"cjk",      "\xe6\x97\xa5\xe6\x9c\x88\xe7\x81\xab\xe6\xb0\xb4\xe6\x9c\xa8\xe9\x87\x91\xe5\x9c\x9f", 1, 0},  /* 日月火水木金土 */
    {"kanji",    "\xe6\x9c\x88\xe7\x81\xab\xe6\xb0\xb4\xe6\x9c\xa8\xe9\x87\x91\xe5\x9c\x9f\xe6\x97\xa5", 1, 0},  /* 月火水木金土日 */
    {"emoji",    "\xf0\x9f\x9f\xa5\xf0\x9f\x9f\xa7\xf0\x9f\x9f\xa8\xf0\x9f\x9f\xa9\xf0\x9f\x9f\xa6\xf0\x9f\x9f\xaa\xe2\xac\x9b", 1, 0},  /* Colored squares */
    {NULL, NULL, 0, 0}
};

/* Border characters */
static const char *B_TOP_L, *B_TOP_R, *B_BOT_L, *B_BOT_R, *B_HORIZ, *B_VERT;
static const char *FILL[8];  /* indexed: 0=empty, 1-7 for cells */
static int cell_width = 2;
static char custom_fill[8][16];

/* UTF-8 helper: get byte length of character */
static int utf8_char_len(const char *s) {
    unsigned char c = (unsigned char)*s;
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

/* Map cell value to array index */
static int cell_idx(int cell) {
    switch(cell) {
        case 1: return 1; case 2: return 2; case 4: return 3;
        case 6: return 4; case 12: return 5; case 15: return 6; case 20: return 7;
    }
    return 0;
}

static void setup_fills(const render_opts_t *opts) {
    int double_fills = (cell_width == 2) && !opts->wide_fills;

    if (opts->fills && *opts->fills) {
        /* Custom fills */
        const char *p = opts->fills;
        for (int i = 0; i < 7 && *p; i++) {
            int len = utf8_char_len(p);
            if (double_fills) {
                memcpy(custom_fill[i+1], p, len);
                memcpy(custom_fill[i+1] + len, p, len);
                custom_fill[i+1][len * 2] = '\0';
            } else {
                memcpy(custom_fill[i+1], p, len);
                custom_fill[i+1][len] = '\0';
            }
            FILL[i+1] = custom_fill[i+1];
            p += len;
        }
    } else {
        /* Default: solid blocks */
        if (double_fills) {
            for (int i = 1; i < 8; i++)
                FILL[i] = "\xe2\x96\x88\xe2\x96\x88";  /* ██ */
        } else {
            for (int i = 1; i < 8; i++)
                FILL[i] = "\xe2\x96\x88";  /* █ */
        }
    }

    /* Empty fill */
    if (opts->half_width) {
        FILL[0] = " ";
    } else {
        FILL[0] = "  ";
    }
}

static void setup_borders(const render_opts_t *opts) {
    cell_width = opts->half_width ? 1 : 2;

    if (opts->ascii) {
        B_TOP_L = "."; B_TOP_R = ".";
        B_BOT_L = "'"; B_BOT_R = "'";
        B_HORIZ = "-"; B_VERT = "|";
    } else {
        B_TOP_L = "\xe2\x94\x8c";  /* ┌ */
        B_TOP_R = "\xe2\x94\x90";  /* ┐ */
        B_BOT_L = "\xe2\x94\x94";  /* └ */
        B_BOT_R = "\xe2\x94\x98";  /* ┘ */
        B_HORIZ = "\xe2\x94\x80";  /* ─ */
        B_VERT  = "\xe2\x94\x82";  /* │ */
    }

    setup_fills(opts);
}

void render_opts_init(render_opts_t *opts) {
    opts->ascii = 0;
    opts->half_width = 0;
    opts->wide_fills = 0;
    opts->fills = NULL;
    opts->preset = NULL;
    /* Apply default preset (cjk) */
    render_apply_preset(opts, "cjk");
}

int render_apply_preset(render_opts_t *opts, const char *preset_name) {
    for (int i = 0; PRESETS[i].name; i++) {
        if (strcmp(preset_name, PRESETS[i].name) == 0) {
            opts->fills = PRESETS[i].chars;
            opts->wide_fills = PRESETS[i].wide;
            opts->half_width = PRESETS[i].half;
            opts->preset = preset_name;
            return 0;
        }
    }
    return -1;
}

int is_visible(uint8_t mask, int cell) {
    switch(cell) {
        case 1:  return mask & 0x01;
        case 2:  return mask & 0x02;
        case 4:  return mask & 0x04;
        case 6:  return mask & 0x08;
        case 12: return mask & 0x10;
        case 15: return mask & 0x20;
        case 20: return mask & 0x40;
    }
    return 0;
}

void render_mask(uint8_t mask, const render_opts_t *opts, FILE *out) {
    setup_borders(opts);

    int border_width = cell_width * 6;

    /* Top border */
    fprintf(out, "%s", B_TOP_L);
    for (int i = 0; i < border_width; i++) fprintf(out, "%s", B_HORIZ);
    fprintf(out, "%s\n", B_TOP_R);

    /* Content rows */
    for (int r = 0; r < 10; r++) {
        fprintf(out, "%s", B_VERT);
        for (int c = 0; c < 6; c++) {
            int cell = GRID[r][c];
            int vis = is_visible(mask, cell);
            int idx = vis ? cell_idx(cell) : 0;
            fprintf(out, "%s", FILL[idx]);
        }
        fprintf(out, "%s\n", B_VERT);
    }

    /* Bottom border */
    fprintf(out, "%s", B_BOT_L);
    for (int i = 0; i < border_width; i++) fprintf(out, "%s", B_HORIZ);
    fprintf(out, "%s\n", B_BOT_R);
}

static char render_buffer[2048];

const char *render_mask_to_str(uint8_t mask, const render_opts_t *opts) {
    setup_borders(opts);

    char *p = render_buffer;
    int border_width = cell_width * 6;

    /* Top border */
    p += sprintf(p, "%s", B_TOP_L);
    for (int i = 0; i < border_width; i++) p += sprintf(p, "%s", B_HORIZ);
    p += sprintf(p, "%s\n", B_TOP_R);

    /* Content rows */
    for (int r = 0; r < 10; r++) {
        p += sprintf(p, "%s", B_VERT);
        for (int c = 0; c < 6; c++) {
            int cell = GRID[r][c];
            int vis = is_visible(mask, cell);
            int idx = vis ? cell_idx(cell) : 0;
            p += sprintf(p, "%s", FILL[idx]);
        }
        p += sprintf(p, "%s\n", B_VERT);
    }

    /* Bottom border */
    p += sprintf(p, "%s", B_BOT_L);
    for (int i = 0; i < border_width; i++) p += sprintf(p, "%s", B_HORIZ);
    p += sprintf(p, "%s\n", B_BOT_R);

    return render_buffer;
}

static const char *preset_names[16];

const char **render_get_presets(void) {
    int i;
    for (i = 0; PRESETS[i].name && i < 15; i++) {
        preset_names[i] = PRESETS[i].name;
    }
    preset_names[i] = NULL;
    return preset_names;
}
