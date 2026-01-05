/* Mondrian Terminal Clock - minimal C implementation */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#define SLEEP_MS(ms) Sleep(ms)
#define IS_TTY() _isatty(_fileno(stdout))
static void enable_utf8(void) { SetConsoleOutputCP(65001); }
static void enable_vt_mode(void) {
    /* Enable ANSI escape sequences on Windows 10+ */
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    if (GetConsoleMode(hOut, &mode)) {
        SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    }
}
static int get_milliseconds(void) {
    SYSTEMTIME st;
    GetSystemTime(&st);
    return st.wMilliseconds;
}
#else
#include <unistd.h>
#include <sys/time.h>
#define SLEEP_MS(ms) usleep((ms)*1000)
#define IS_TTY() isatty(STDOUT_FILENO)
static void enable_utf8(void) { /* UTF-8 default on Unix */ }
static void enable_vt_mode(void) { /* ANSI supported natively on Unix */ }
static int get_milliseconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int)(tv.tv_usec / 1000);
}
#endif

static volatile int running = 1;

static void handle_signal(int sig) {
    (void)sig;
    running = 0;
}

/* Cell ownership for 6x10 grid (row-major) */
static const int GRID[10][6] = {
    {20,20,20,20,12,12}, {20,20,20,20,12,12}, {20,20,20,20,12,12},
    {20,20,20,20,12,12}, {20,20,20,20,12,12}, {15,15,15,1,12,12},
    {15,15,15,2,4,4}, {15,15,15,2,4,4}, {15,15,15,6,6,6}, {15,15,15,6,6,6}
};

/* Combinations for each second - stored as bitmask (bits: 20,15,12,6,4,2,1) */
static const uint8_t COMBOS[60][4] = {
    {0x00,0x7F}, {0x01}, {0x02}, {0x03}, {0x04}, {0x05}, {0x08,0x06}, {0x09,0x07},
    {0x0A}, {0x0B}, {0x0C}, {0x0D}, {0x10,0x0E}, {0x11,0x0F}, {0x12}, {0x20,0x13},
    {0x21,0x14}, {0x22,0x15}, {0x23,0x18,0x16}, {0x24,0x19,0x17}, {0x40,0x25,0x1A},
    {0x41,0x28,0x26,0x1B}, {0x42,0x29,0x27,0x1C}, {0x43,0x2A,0x1D}, {0x44,0x2B,0x1E},
    {0x45,0x2C,0x1F}, {0x48,0x46,0x2D}, {0x49,0x47,0x30,0x2E}, {0x31,0x4A,0x2F},
    {0x4B,0x32}, {0x33,0x4C}, {0x4D,0x34}, {0x50,0x35,0x4E}, {0x51,0x4F,0x38,0x36},
    {0x52,0x39,0x37}, {0x60,0x53,0x3A}, {0x61,0x54,0x3B}, {0x62,0x55,0x3C},
    {0x63,0x58,0x56,0x3D}, {0x64,0x59,0x57,0x3E}, {0x65,0x5A,0x3F}, {0x68,0x66,0x5B},
    {0x69,0x67,0x5C}, {0x6A,0x5D}, {0x6B,0x5E}, {0x6C,0x5F}, {0x6D}, {0x70,0x6E},
    {0x71,0x6F}, {0x72}, {0x73}, {0x74}, {0x75}, {0x78,0x76}, {0x79,0x77}, {0x7A},
    {0x7B}, {0x7C}, {0x7D}, {0x7E}
};
static const uint8_t COMBO_CNT[60] = {
    2,1,1,1,1,1,2,2,1,1,1,1,2,2,1,2,2,2,3,3,3,4,4,3,3,3,3,4,3,2,2,2,3,4,3,3,3,3,4,4,3,3,3,2,2,2,1,2,2,1,1,1,1,2,2,1,1,1,1,1
};

/* Convert bitmask to visible set - returns 1 if cell is visible */
static int is_visible(uint8_t mask, int cell) {
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

/* Compute permutation index for second s given minute k */
static int perm_index(uint64_t k, int s) {
    static const uint64_t PERIOD = 46221064723759104ULL;
    k = k % PERIOD;

    int m = COMBO_CNT[s];
    if (m == 1) return 0;

    /* Count how many 2/3/4-option seconds come before s */
    int n2 = 0, n3 = 0, n4 = 0;
    for (int i = 0; i < s; i++) {
        if (COMBO_CNT[i] == 2) n2++;
        else if (COMBO_CNT[i] == 3) n3++;
        else if (COMBO_CNT[i] == 4) n4++;
    }

    uint64_t k3_val = k / 1073741824ULL;
    uint64_t rest = k % 1073741824ULL;

    if (m == 2) return (rest >> (29 - n2)) & 1;
    if (m == 3) {
        uint64_t p = 1;
        for (int i = 0; i < 15 - n3; i++) p *= 3;
        return (k3_val / p) % 3;
    }
    if (m == 4) return (rest >> (10 - 2*n4)) & 3;
    return 0;
}

/* Border and fill characters */
static const char *B_TOP_L, *B_TOP_R, *B_BOT_L, *B_BOT_R, *B_HORIZ, *B_VERT;
static const char *FILL[8]; /* indexed: 0=empty, 1,2,4,6,12,15,20 mapped */
static int cell_width = 2;  /* columns per cell: 2 (default) or 1 (half-width) */

/* Map cell value to index: 1->1, 2->2, 4->3, 6->4, 12->5, 15->6, 20->7 */
static int cell_idx(int cell) {
    switch(cell) {
        case 1: return 1; case 2: return 2; case 4: return 3;
        case 6: return 4; case 12: return 5; case 15: return 6; case 20: return 7;
    }
    return 0;
}

/*
 * Graph coloring: no two edge-adjacent cells share the same fill.
 * Edge adjacencies: 20-{12,15,1}, 12-{20,1,4}, 15-{20,1,2,6}, 1-{20,12,15,2},
 *                   2-{15,1,4,6}, 4-{12,2,6}, 6-{15,2,4}
 * 4-coloring required (3 is insufficient):
 *   A: 4, 15    B: 2, 20    C: 1    D: 6, 12
 */
static char custom_fill[8][9];  /* For -f option: 2 UTF-8 chars (up to 4 bytes each) + null */

/* UTF-8 helper: get byte length of character starting at s */
static int utf8_char_len(const char *s) {
    unsigned char c = (unsigned char)*s;
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

static void set_ascii(int distinct, const char *fill_chars, int half_width, int wide_fills) {
    B_TOP_L = "."; B_TOP_R = "."; B_BOT_L = "'"; B_BOT_R = "'";
    B_HORIZ = "-"; B_VERT = "|";

    /* Set cell width and empty fill */
    if (half_width) {
        cell_width = 1;
        FILL[0] = " ";
    } else {
        cell_width = 2;
        FILL[0] = "  ";
    }

    /* Double fills only when: 2-column cells AND not using wide glyphs */
    int double_fills = (cell_width == 2) && !wide_fills;

    if (fill_chars && *fill_chars) {
        /* Custom: 7 UTF-8 glyphs for cells 1,2,4,6,12,15,20 */
        const char *p = fill_chars;
        int has_space = 0;
        for (int i = 0; i < 7 && *p; i++) {
            int len = utf8_char_len(p);
            if (*p == ' ') has_space = 1;
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
        if (has_space) {
            fprintf(stderr, "Warning: -f contains space - inverse mode won't work\n");
        }
    } else if (distinct) {
        /* 4-color graph coloring */
        if (double_fills) {
            FILL[1] = "$$"; FILL[2] = "@@"; FILL[3] = "##";
            FILL[4] = "%%"; FILL[5] = "%%"; FILL[6] = "##"; FILL[7] = "@@";
        } else {
            FILL[1] = "$"; FILL[2] = "@"; FILL[3] = "#";
            FILL[4] = "%"; FILL[5] = "%"; FILL[6] = "#"; FILL[7] = "@";
        }
    } else {
        for (int i = 1; i < 8; i++) FILL[i] = double_fills ? "##" : "#";
    }
}

static void set_unicode(int distinct, const char *fill_chars, int half_width, int wide_fills) {
    B_TOP_L = "\xe2\x94\x8c"; B_TOP_R = "\xe2\x94\x90";
    B_BOT_L = "\xe2\x94\x94"; B_BOT_R = "\xe2\x94\x98";
    B_HORIZ = "\xe2\x94\x80"; B_VERT = "\xe2\x94\x82";

    /* Set cell width and empty fill */
    if (half_width) {
        cell_width = 1;
        FILL[0] = " ";
    } else {
        cell_width = 2;
        FILL[0] = "  ";
    }

    /* Double fills only when: 2-column cells AND not using wide glyphs */
    int double_fills = (cell_width == 2) && !wide_fills;

    if (fill_chars && *fill_chars) {
        /* Custom: 7 UTF-8 glyphs for cells 1,2,4,6,12,15,20 */
        const char *p = fill_chars;
        int has_space = 0;
        for (int i = 0; i < 7 && *p; i++) {
            int len = utf8_char_len(p);
            if (*p == ' ') has_space = 1;
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
        if (has_space) {
            fprintf(stderr, "Warning: -f contains space - inverse mode won't work\n");
        }
    } else if (distinct) {
        /* 4-color graph coloring: A=██, B=▓▓, C=▒▒, D=░░ */
        if (double_fills) {
            FILL[1] = "\xe2\x96\x92\xe2\x96\x92"; /* ▒▒ */
            FILL[2] = "\xe2\x96\x93\xe2\x96\x93"; /* ▓▓ */
            FILL[3] = "\xe2\x96\x88\xe2\x96\x88"; /* ██ */
            FILL[4] = "\xe2\x96\x91\xe2\x96\x91"; /* ░░ */
            FILL[5] = "\xe2\x96\x91\xe2\x96\x91"; /* ░░ */
            FILL[6] = "\xe2\x96\x88\xe2\x96\x88"; /* ██ */
            FILL[7] = "\xe2\x96\x93\xe2\x96\x93"; /* ▓▓ */
        } else {
            FILL[1] = "\xe2\x96\x92"; /* ▒ */
            FILL[2] = "\xe2\x96\x93"; /* ▓ */
            FILL[3] = "\xe2\x96\x88"; /* █ */
            FILL[4] = "\xe2\x96\x91"; /* ░ */
            FILL[5] = "\xe2\x96\x91"; /* ░ */
            FILL[6] = "\xe2\x96\x88"; /* █ */
            FILL[7] = "\xe2\x96\x93"; /* ▓ */
        }
    } else {
        for (int i = 1; i < 8; i++)
            FILL[i] = double_fills ? "\xe2\x96\x88\xe2\x96\x88" : "\xe2\x96\x88";
    }
}

static void render(uint8_t mask) {
    int border_width = cell_width * 6;  /* 6 cells × cell_width columns */

    /* Top border */
    printf("%s", B_TOP_L);
    for (int i = 0; i < border_width; i++) printf("%s", B_HORIZ);
    printf("%s\n", B_TOP_R);

    /* 10 content rows */
    for (int r = 0; r < 10; r++) {
        printf("%s", B_VERT);
        for (int c = 0; c < 6; c++) {
            int cell = GRID[r][c];
            int vis = is_visible(mask, cell);
            int idx = vis ? cell_idx(cell) : 0;
            printf("%s", FILL[idx]);
        }
        printf("%s\n", B_VERT);
    }

    /* Bottom border */
    printf("%s", B_BOT_L);
    for (int i = 0; i < border_width; i++) printf("%s", B_HORIZ);
    printf("%s\n\n", B_BOT_R);
    fflush(stdout);
}

/* Parse ISO 8601 date: YYYY-MM-DDTHH:MM:SSZ or YYYY-MM-DD */
static time_t parse_origin(const char *s) {
    struct tm tm = {0};
    int y, m, d, H = 0, M = 0, S = 0;
    if (sscanf(s, "%d-%d-%dT%d:%d:%d", &y, &m, &d, &H, &M, &S) >= 3 ||
        sscanf(s, "%d-%d-%d", &y, &m, &d) == 3) {
        tm.tm_year = y - 1900; tm.tm_mon = m - 1; tm.tm_mday = d;
        tm.tm_hour = H; tm.tm_min = M; tm.tm_sec = S;
#ifdef _WIN32
        return _mkgmtime(&tm);
#else
        return timegm(&tm);
#endif
    }
    return 0; /* Unix epoch as fallback */
}

/*
 * ==================== INVERSE MODE ====================
 * Parse frames from stdin, detect filled cells, reconstruct k
 */

/* Check if position has a fill character (handles UTF-8) */
static int is_fill_at(const char *s, int pos) {
    unsigned char c = (unsigned char)s[pos];
    /* Empty and structural characters */
    if (c == ' ' || c == '|' || c == '-' || c == '.' || c == '\'' ||
        c == '\n' || c == '\r' || c == '\0') return 0;
    /* UTF-8 box drawing (E2 94 xx) is border, not fill */
    if (c == 0xe2 && (unsigned char)s[pos+1] == 0x94) return 0;
    /* Any other UTF-8 multi-byte sequence is fill */
    if (c >= 0x80) return 1;
    /* ASCII fills: any printable except structural chars */
    return (c >= 0x21 && c <= 0x7e);
}

/* Count dashes in border line to detect half-width mode */
static int count_border_dashes(const char *line) {
    int count = 0;
    for (const char *p = line; *p; p++) {
        if (*p == '-') count++;
        /* UTF-8 horizontal line ─ (E2 94 80) */
        if ((unsigned char)*p == 0xe2 && (unsigned char)*(p+1) == 0x94 &&
            (unsigned char)*(p+2) == 0x80) {
            count++;
            p += 2;
        }
    }
    return count;
}

/* Count Unicode characters in content area (between borders, excluding borders) */
static int get_content_chars(const char *line) {
    const char *p = line;
    /* Skip left border */
    if ((unsigned char)*p == 0xe2) p += 3;
    else if (*p == '|') p += 1;
    int count = 0;
    /* Count chars until right border */
    while (*p && *p != '\n' && *p != '\r') {
        if (*p == '|') break;
        if ((unsigned char)*p == 0xe2 && (unsigned char)*(p+1) == 0x94 &&
            (unsigned char)*(p+2) == 0x82) break;  /* │ */
        p += utf8_char_len(p);
        count++;
    }
    return count;
}

/* Buffer for last non-frame line (used for timestamp parsing) */
static char last_nonframe_line[256] = {0};

/* Parse a single frame (12+ lines), return bitmask of visible cells, or -1 on EOF/error */
static int parse_frame(FILE *f) {
    char line[256];
    int cells_visible[8] = {0}; /* indexed by cell_idx */
    int content_row = 0;
    int half_width_mode = 0;
    int wide_mode = 0;

    /* Skip until we find a top border line (┌ or . followed by dashes) */
    while (fgets(line, sizeof(line), f)) {
        unsigned char c0 = (unsigned char)line[0];
        unsigned char c1 = (unsigned char)line[1];
        unsigned char c2 = (unsigned char)line[2];
        /* ASCII top-left corner '.' or Unicode ┌ (\xe2\x94\x8c) */
        if (c0 == '.' || (c0 == 0xe2 && c1 == 0x94 && c2 == 0x8c)) break;
        /* Save non-empty lines for potential timestamp parsing */
        if (c0 >= '0' && c0 <= '9') strncpy(last_nonframe_line, line, 255);
        if (feof(f)) return -1;
    }
    if (feof(f)) return -1;

    /* Detect half-width mode from border: 6 dashes = half, 12 = normal */
    int dashes = count_border_dashes(line);
    half_width_mode = (dashes <= 8);  /* Allow some tolerance */

    /* Store content lines to allow pre-scan for mode detection */
    char content_lines[10][256];
    int num_content_lines = 0;

    /* Read all content rows first */
    while (num_content_lines < 10 && fgets(line, sizeof(line), f)) {
        if (line[0] == '\n' || line[0] == '\r') continue;
        unsigned char c0 = (unsigned char)line[0];
        unsigned char c1 = (unsigned char)line[1];
        unsigned char c2 = (unsigned char)line[2];
        if (c0 == '.' || c0 == '\'') break;
        if (c0 == 0xe2 && c1 == 0x94 && (c2 == 0x8c || c2 == 0x94 || c2 == 0x9c)) break;
        if (strchr(line, '-') && !strchr(line, '#') && !strchr(line, '@') &&
            !strchr(line, '%') && !strchr(line, ':') &&
            !strstr(line, "\xe2\x96")) continue;
        strncpy(content_lines[num_content_lines], line, 255);
        content_lines[num_content_lines][255] = '\0';
        num_content_lines++;
    }

    /* Detect wide mode vs doubled mode using character count.
     * In doubled mode, fills are printed twice (e.g., "██"), run lengths always even.
     * In wide mode, fills are single wide chars, run lengths can be odd. */
    if (!half_width_mode) {
        int max_chars = 0;
        for (int i = 0; i < num_content_lines; i++) {
            int chars = get_content_chars(content_lines[i]);
            if (chars > max_chars) max_chars = chars;
        }
        if (max_chars > 12) {
            wide_mode = 0;  /* More than 12 chars = definitely doubled */
        } else {
            /* Check for odd-length runs of identical characters.
             * In doubled mode: all runs are even (pairs).
             * In wide mode: runs can be odd. */
            int found_odd_run = 0;
            for (int i = 0; i < num_content_lines && !found_odd_run; i++) {
                const char *p = content_lines[i];
                if ((unsigned char)*p == 0xe2) p += 3;
                else if (*p == '|') p += 1;
                while (*p && *p != '\n' && !found_odd_run) {
                    unsigned char c = (unsigned char)*p;
                    if (c == ' ' || c == '|') { p++; continue; }
                    if (c == 0xe2 && (unsigned char)*(p+1) == 0x94) { p += 3; continue; }
                    /* Count consecutive identical characters */
                    int char_len = utf8_char_len(p);
                    int run_len = 0;
                    while (memcmp(p, p + run_len * char_len, char_len) == 0 &&
                           *(p + run_len * char_len) != '\0')
                        run_len++;
                    if (run_len % 2 == 1) found_odd_run = 1;
                    p += run_len * char_len;
                }
            }
            wide_mode = found_odd_run;
        }
    }

    /* Now parse the stored content rows */
    for (content_row = 0; content_row < num_content_lines; content_row++) {
        char *cline = content_lines[content_row];
        int pos = 0;
        /* Skip left border (ASCII | or UTF-8 │) */
        if ((unsigned char)cline[pos] == 0xe2) pos += 3;
        else if (cline[pos] == '|') pos += 1;

        for (int c = 0; c < 6 && cline[pos]; c++) {
            int filled = is_fill_at(cline, pos);

            /* Advance based on mode and character type */
            unsigned char ch = (unsigned char)cline[pos];
            int advance;

            if (half_width_mode) {
                /* Half-width: 1 column per cell */
                advance = (ch >= 0x80) ? utf8_char_len(cline + pos) : 1;
            } else if (wide_mode) {
                /* Wide: 1 fill char per cell, but 2 spaces for empty */
                if (ch == ' ') advance = 2;
                else advance = (ch >= 0x80) ? utf8_char_len(cline + pos) : 1;
            } else {
                /* Normal: 2 characters per cell (doubled fills) */
                if (ch >= 0x80) {
                    advance = utf8_char_len(cline + pos) * 2;
                } else if (ch == ' ') {
                    advance = 2;
                } else {
                    advance = 2;  /* Doubled ASCII */
                }
            }
            pos += advance;

            /* Skip potential grid dividers */
            while (cline[pos] == '|' || ((unsigned char)cline[pos] == 0xe2 &&
                   (unsigned char)cline[pos+1] == 0x94)) {
                if ((unsigned char)cline[pos] == 0xe2) pos += 3;
                else pos++;
            }

            if (filled) {
                int cell = GRID[content_row][c];
                cells_visible[cell_idx(cell)] = 1;
            }
        }
    }

    /* Skip to empty line (frame separator) */
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '\n' || line[0] == '\r' || strlen(line) <= 1) break;
    }

    if (content_row < 10) return -1;

    /* Return bitmask: bits 0-6 for cells 1,2,4,6,12,15,20 */
    int mask = 0;
    if (cells_visible[1]) mask |= 0x01; /* 1 */
    if (cells_visible[2]) mask |= 0x02; /* 2 */
    if (cells_visible[3]) mask |= 0x04; /* 4 */
    if (cells_visible[4]) mask |= 0x08; /* 6 */
    if (cells_visible[5]) mask |= 0x10; /* 12 */
    if (cells_visible[6]) mask |= 0x20; /* 15 */
    if (cells_visible[7]) mask |= 0x40; /* 20 */

    /* Validate: need exactly 10 content rows for complete frame */
    if (num_content_lines != 10) return -1;

    return mask;
}

/* Convert mask to sum */
static int mask_to_sum(int mask) {
    int sum = 0;
    if (mask & 0x01) sum += 1;
    if (mask & 0x02) sum += 2;
    if (mask & 0x04) sum += 4;
    if (mask & 0x08) sum += 6;
    if (mask & 0x10) sum += 12;
    if (mask & 0x20) sum += 15;
    if (mask & 0x40) sum += 20;
    return sum;
}

/* Find which combination index produces the given mask for second s */
static int find_combo_idx(int s, uint8_t mask) {
    for (int i = 0; i < COMBO_CNT[s]; i++) {
        if (COMBOS[s][i] == mask) return i;
    }
    return -1;
}

static const uint64_t PERIOD = 46221064723759104ULL;

/* Verify k against observations spanning two minutes.
 * When first_sec > 0: seconds 0..first_sec-1 from minute k,
 *                     seconds first_sec..59 from minute k-1.
 * Returns match count (0-60). */
static int verify_k_spanning(uint64_t k, uint8_t masks[60], int rot, int first_sec) {
    uint64_t k_minus_1 = (k > 0) ? k - 1 : PERIOD - 1;
    int matches = 0;

    for (int s = 0; s < 60; s++) {
        int frame_idx = (s + rot) % 60;
        uint64_t minute;
        if (first_sec == 0) {
            minute = k;
        } else if (s < first_sec) {
            minute = k;         /* Seconds 0..first_sec-1 from minute k */
        } else {
            minute = k_minus_1; /* Seconds first_sec..59 from minute k-1 */
        }
        int expected_idx = perm_index(minute, s);
        uint8_t expected_mask = COMBOS[s][expected_idx];
        if (expected_mask == masks[frame_idx]) matches++;
    }
    return matches;
}

/* Reconstruct k from 60 observed masks */
static int reconstruct_k(uint8_t masks[60], uint64_t *out_k) {
    /* Try all 60 rotations to find the starting second */
    for (int rot = 0; rot < 60; rot++) {
        uint64_t k2_bits = 0, k4_bits = 0;
        int k3_digits[16];
        int n2 = 0, n3 = 0, n4 = 0;
        int valid = 1;

        for (int s = 0; s < 60 && valid; s++) {
            int frame_idx = (s + rot) % 60;
            uint8_t mask = masks[frame_idx];
            int sum = mask_to_sum(mask) % 60;  /* mod 60: second 0 can be 0 or 60 */

            /* The sum tells us which second this is */
            if (sum != s) { valid = 0; break; }

            /* Find which combo index was used */
            int idx = find_combo_idx(s, mask);
            if (idx < 0) { valid = 0; break; }

            /* Extract choice bits based on how many options this second has */
            int m = COMBO_CNT[s];
            if (m == 1) continue;

            if (m == 2) {
                k2_bits |= ((uint64_t)idx << (29 - n2));
                n2++;
            } else if (m == 3) {
                if (n3 < 16) k3_digits[n3++] = idx;
            } else if (m == 4) {
                k4_bits |= ((uint64_t)idx << (10 - 2*n4));
                n4++;
            }
        }

        if (!valid) continue;

        /* Reconstruct k from extracted bits */
        uint64_t k3_val = 0;
        for (int i = 0; i < n3; i++) {
            k3_val = k3_val * 3 + k3_digits[i];
        }
        uint64_t k_candidate = k3_val * 1073741824ULL + (k2_bits | k4_bits);
        int first_sec = (rot == 0) ? 0 : (60 - rot);

        /* When first_sec > 0, observations span two minutes.
         * Search around k_candidate and also around 0 (for wrap-around). */
        uint64_t best_k = 0;
        int best_matches = 0;

        uint64_t search_centers[2] = {k_candidate, 0};
        for (int c = 0; c < 2; c++) {
            uint64_t center = search_centers[c];
            for (int delta = 0; delta < 100; delta++) {
                for (int sign = 0; sign <= 1; sign++) {
                    uint64_t candidate;
                    if (sign == 0) {
                        candidate = (center + delta) % PERIOD;
                    } else {
                        candidate = (center >= (uint64_t)delta) ? center - delta : PERIOD - delta + center;
                    }
                    int matches = verify_k_spanning(candidate, masks, rot, first_sec);
                    if (matches > best_matches) {
                        best_matches = matches;
                        best_k = candidate;
                    }
                    /* Perfect match - return immediately */
                    if (matches == 60) {
                        *out_k = candidate;
                        return rot;
                    }
                }
            }
        }

        /* Return best match if good enough (90% = 54/60) */
        if (best_matches >= 54) {
            *out_k = best_k;
            return rot;
        }
    }
    return -1; /* Failed */
}

static int run_inverse(void) {
    uint8_t masks[60];
    int total = 0;  /* Total frames read */
    last_nonframe_line[0] = '\0';  /* Reset buffer */

    fprintf(stderr, "Reading frames from stdin...\n");

    /* Read all frames, keeping last 60 in circular buffer */
    while (1) {
        int mask = parse_frame(stdin);
        if (mask < 0) break;
        masks[total % 60] = (uint8_t)mask;
        int sum = mask_to_sum(mask) % 60;
        fprintf(stderr, "\rFrame %d: sum=%d  ", total + 1, sum);
        total++;
    }
    fprintf(stderr, "\n");

    if (total < 60) {
        fprintf(stderr, "Error: Need 60 frames, got %d\n", total);
        return 1;
    }

    /* Rotate buffer so masks[0] is the first of the last 60 frames */
    if (total > 60) {
        uint8_t temp[60];
        int start = total % 60;
        for (int i = 0; i < 60; i++)
            temp[i] = masks[(start + i) % 60];
        memcpy(masks, temp, 60);
        fprintf(stderr, "Using last 60 of %d frames\n", total);
    }

    /* Try to read timestamp line: ISO8601 format */
    /* First check the buffer from parse_frame (in case timestamp was consumed) */
    time_t start_time = 0;
    if (last_nonframe_line[0] >= '0' && last_nonframe_line[0] <= '9' &&
        strlen(last_nonframe_line) >= 20) {
        start_time = parse_origin(last_nonframe_line);
    }
    /* Then try reading more lines */
    if (start_time == 0) {
        char line[256];
        while (fgets(line, sizeof(line), stdin)) {
            if (line[0] >= '0' && line[0] <= '9' && strlen(line) >= 20) {
                start_time = parse_origin(line);
                if (start_time > 0) break;
            }
        }
    }
    if (start_time == 0) {
        fprintf(stderr, "Warning: No timestamp found, origin will be invalid\n");
        start_time = time(NULL);
    }

    fprintf(stderr, "Reconstructing...\n");
    uint64_t k;
    int rot = reconstruct_k(masks, &k);
    if (rot < 0) {
        fprintf(stderr, "Error: Could not reconstruct k\n");
        return 1;
    }

    /* First frame's second = (60 - rot) % 60 */
    int first_sec = (rot == 0) ? 0 : (60 - rot);

    /* k is the minute where second 0 appears. If first_sec > 0, the first
     * frame is actually in minute k-1 (spanning the minute boundary). */
    uint64_t first_min = (first_sec > 0) ? k - 1 : k;
    uint64_t t = first_min * 60 + (uint64_t)first_sec;
    fprintf(stderr, "t = %llu\n", (unsigned long long)t);

    /* Compute origin: timestamp is for the LAST (60th) frame.
     * First frame time = last frame time - 59 seconds.
     * Then find when second 0 occurred and subtract k minutes. */
    time_t first_frame_time = start_time - 59;
    int seconds_to_second_0 = (60 - first_sec) % 60;
    time_t second_0_time = first_frame_time + seconds_to_second_0;
    time_t origin_time = second_0_time - (time_t)(k * 60);
    struct tm *utc = gmtime(&origin_time);

    /* Output origin as ISO 8601 UTC */
    printf("%04d-%02d-%02dT%02d:%02d:%02dZ\n",
           utc->tm_year + 1900, utc->tm_mon + 1, utc->tm_mday,
           utc->tm_hour, utc->tm_min, utc->tm_sec);
    return 0;
}

static void usage(const char *prog) {
    printf("Mondrian Terminal Clock\n\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Modes:\n");
    printf("  (default)   Live clock, display frames (1/sec)\n");
    printf("  -s          Simulate: fast output (no delay), timestamp as-if-live\n");
    printf("  -l          In-place update (no scroll, requires TTY)\n");
    printf("  -i          Inverse: read frames from stdin, output k and origin\n");
    printf("  -n N        Output N frames then exit\n\n");
    printf("Display:\n");
    printf("  -a          ASCII borders (.|'-)\n");
    printf("  -u          Unicode borders (box drawing) [default]\n");
    printf("  -p PRESET   Preset: cjk [default], blocks, blocks1, distinct, kanji, emoji\n");
    printf("  -f CHARS    Custom 7 UTF-8 fill characters (one per cell type)\n");
    printf("  -1          Half-width: 1 column per cell (compact 8-col output)\n");
    printf("  -w          Wide fills: -f glyphs are full-width (CJK, 2 cols each)\n\n");
    printf("Time:\n");
    printf("  -o ORIGIN   Custom origin (ISO 8601, e.g. 2000-01-01T00:00:00Z)\n");
    printf("  -t T        Use absolute time T seconds from origin\n\n");
    printf("Examples:\n");
    printf("  %s                    # Live clock (CJK default)\n", prog);
    printf("  %s -l                 # In-place update\n", prog);
    printf("  %s -p emoji -l        # Colored squares\n", prog);
    printf("  %s -p blocks          # Classic monochrome\n", prog);
    printf("  %s -n 60 -s | %s -i   # Round-trip test\n", prog, prog);
}

/* Fill presets: name, chars, wide, half */
static struct { const char *name; const char *chars; int wide; int half; } PRESETS[] = {
    {"blocks",   "███████", 0, 0},       /* Default monochrome */
    {"blocks1",  "███████", 0, 1},       /* Half-width monochrome */
    {"distinct", "▒▓█░░█▓", 0, 0},       /* 4-color shading */
    {"cjk",      "日月火水木金土", 1, 0}, /* Elements (wide) */
    {"kanji",    "月火水木金土日", 1, 0}, /* Days of week (wide) */
    {"emoji",    "🟥🟧🟨🟩🟦🟪⬛", 1, 0}, /* Colored squares (wide) */
    {NULL, NULL, 0, 0}
};

int main(int argc, char **argv) {
    int unicode = 1, inverse = 0, live_inplace = 0, simulate = 0;
    int half_width = 0, wide_fills = 0;
    time_t origin = 0;  /* Unix epoch default, same as webpage */
    int64_t fixed_t = -1;  /* -1 means use system time, else seconds from origin */
    int64_t num_frames = -1;  /* -1 means infinite (live mode) */
    const char *preset = NULL;
    const char *fill_chars = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-a") == 0) unicode = 0;
        else if (strcmp(argv[i], "-u") == 0) unicode = 1;
        else if (strcmp(argv[i], "-1") == 0) half_width = 1;
        else if (strcmp(argv[i], "-w") == 0) wide_fills = 1;
        else if (strcmp(argv[i], "-l") == 0) live_inplace = 1;
        else if (strcmp(argv[i], "-s") == 0) simulate = 1;
        else if (strcmp(argv[i], "-i") == 0) inverse = 1;
        else if (strcmp(argv[i], "-p") == 0 && i+1 < argc)
            preset = argv[++i];
        else if (strcmp(argv[i], "-f") == 0 && i+1 < argc)
            fill_chars = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc)
            origin = parse_origin(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i+1 < argc)
            fixed_t = atoll(argv[++i]);
        else if (strcmp(argv[i], "-n") == 0 && i+1 < argc)
            num_frames = atoll(argv[++i]);
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]); return 0;
        }
    }

    if (inverse) return run_inverse();

    /* Apply preset (default: cjk) - sets fill_chars, wide_fills, half_width */
    if (!preset && !fill_chars) preset = "cjk";
    if (preset) {
        for (int i = 0; PRESETS[i].name; i++) {
            if (strcmp(preset, PRESETS[i].name) == 0) {
                fill_chars = PRESETS[i].chars;
                if (PRESETS[i].wide) wide_fills = 1;
                if (PRESETS[i].half) half_width = 1;
                break;
            }
        }
    }

    enable_vt_mode();  /* Enable ANSI escape sequences (Windows 10+) */
    if (unicode) { enable_utf8(); set_unicode(0, fill_chars, half_width, wide_fills); }
    else { set_ascii(0, fill_chars, half_width, wide_fills); }

    signal(SIGINT, handle_signal);
#ifndef _WIN32
    signal(SIGTERM, handle_signal);
#endif

    time_t start_time = time(NULL);
    int64_t start_elapsed;
    if (fixed_t >= 0) {
        /* -k sets absolute seconds from origin */
        start_elapsed = fixed_t;
    } else {
        start_elapsed = start_time - origin;
    }

    uint64_t frame = 0;
    while (running && (num_frames < 0 || frame < (uint64_t)num_frames)) {
        int64_t virtual_elapsed;
        uint64_t k;
        int sec;

        if (simulate) {
            /* Simulation: advance virtual time by frame number */
            virtual_elapsed = start_elapsed + (int64_t)frame;
        } else if (fixed_t >= 0) {
            /* Live with -k: use fixed starting point + real elapsed */
            virtual_elapsed = start_elapsed + (time(NULL) - start_time);
        } else {
            /* Live: use actual current time */
            virtual_elapsed = time(NULL) - origin;
        }

        k = (uint64_t)(virtual_elapsed / 60);
        sec = (int)(virtual_elapsed % 60);

        int idx = perm_index(k, sec);
        uint8_t mask = COMBOS[sec][idx];

        /* In-place mode: move cursor up to overwrite previous frame (TTY only) */
        if (live_inplace && frame > 0 && IS_TTY()) printf("\033[13A");

        render(mask);

        frame++;
        if (!simulate) {
            /* Sleep until next second (efficient, no drift) */
            time_t current = time(NULL);
            do {
                SLEEP_MS(1000 - get_milliseconds());
            } while (time(NULL) == current && running);
        }
    }

    /* Output termination timestamp for inverse mode (ISO 8601 UTC) */
    /* With -k: origin + virtual_elapsed, otherwise: real time */
    printf("\n");
    time_t end_time;
    if (fixed_t >= 0) {
        /* Virtual time based on fixed start + elapsed frames */
        end_time = origin + (time_t)(start_elapsed + (int64_t)(frame - 1));
    } else if (simulate) {
        end_time = start_time + (time_t)(frame - 1);
    } else {
        end_time = time(NULL);
    }
    struct tm *ts_utc = gmtime(&end_time);
    printf("%04d-%02d-%02dT%02d:%02d:%02dZ\n",
           ts_utc->tm_year + 1900, ts_utc->tm_mon + 1, ts_utc->tm_mday,
           ts_utc->tm_hour, ts_utc->tm_min, ts_utc->tm_sec);

    return 0;
}
