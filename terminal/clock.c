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
#else
#include <unistd.h>
#define SLEEP_MS(ms) usleep((ms)*1000)
#define IS_TTY() isatty(STDOUT_FILENO)
static void enable_utf8(void) { /* UTF-8 default on Unix */ }
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
 * (Corner touching is OK - like Mondrian)
 * Edge adjacencies: 20-{12,15}, 12-{1,4}, 15-{1,2,6}, 1-{2}, 2-{4,6}, 4-{6}
 * 3-coloring solution:
 *   A: 2, 12    B: 4, 15    C: 1, 6, 20
 */
static char custom_fill[8][3];  /* For -f option */

static void set_ascii(int distinct, const char *fill_chars) {
    B_TOP_L = "."; B_TOP_R = "."; B_BOT_L = "'"; B_BOT_R = "'";
    B_HORIZ = "-"; B_VERT = "|";
    FILL[0] = "  ";
    if (fill_chars && strlen(fill_chars) >= 7) {
        /* Custom: 7 chars for cells 1,2,4,6,12,15,20 (in that order) */
        int has_space = 0;
        for (int i = 0; i < 7; i++) {
            if (fill_chars[i] == ' ') has_space = 1;
            custom_fill[i+1][0] = fill_chars[i];
            custom_fill[i+1][1] = fill_chars[i];
            custom_fill[i+1][2] = '\0';
            FILL[i+1] = custom_fill[i+1];
        }
        if (has_space) {
            fprintf(stderr, "Warning: -f contains space - inverse mode won't work\n");
        }
    } else if (distinct) {
        /* 3-color graph coloring: A=##, B=@@, C=%% */
        FILL[1] = "%%"; /* 1 -> C */
        FILL[2] = "##"; /* 2 -> A */
        FILL[3] = "@@"; /* 4 -> B */
        FILL[4] = "%%"; /* 6 -> C */
        FILL[5] = "##"; /* 12 -> A */
        FILL[6] = "@@"; /* 15 -> B */
        FILL[7] = "%%"; /* 20 -> C */
    } else {
        for (int i = 1; i < 8; i++) FILL[i] = "##";
    }
}

static void set_unicode(int distinct) {
    B_TOP_L = "\xe2\x94\x8c"; B_TOP_R = "\xe2\x94\x90";
    B_BOT_L = "\xe2\x94\x94"; B_BOT_R = "\xe2\x94\x98";
    B_HORIZ = "\xe2\x94\x80"; B_VERT = "\xe2\x94\x82";
    FILL[0] = "  ";
    if (distinct) {
        /* 3-color: A=██, B=▓▓, C=▒▒ */
        FILL[1] = "\xe2\x96\x92\xe2\x96\x92"; /* 1 -> C ▒▒ */
        FILL[2] = "\xe2\x96\x88\xe2\x96\x88"; /* 2 -> A ██ */
        FILL[3] = "\xe2\x96\x93\xe2\x96\x93"; /* 4 -> B ▓▓ */
        FILL[4] = "\xe2\x96\x92\xe2\x96\x92"; /* 6 -> C ▒▒ */
        FILL[5] = "\xe2\x96\x88\xe2\x96\x88"; /* 12 -> A ██ */
        FILL[6] = "\xe2\x96\x93\xe2\x96\x93"; /* 15 -> B ▓▓ */
        FILL[7] = "\xe2\x96\x92\xe2\x96\x92"; /* 20 -> C ▒▒ */
    } else {
        for (int i = 1; i < 8; i++) FILL[i] = "\xe2\x96\x88\xe2\x96\x88";
    }
}

static void render(uint8_t mask) {
    /* Top border */
    printf("%s", B_TOP_L);
    for (int i = 0; i < 12; i++) printf("%s", B_HORIZ);
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
    for (int i = 0; i < 12; i++) printf("%s", B_HORIZ);
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
    if (c == ' ' || c == '|' || c == '-' || c == '.' || c == '\'' ||
        c == '\n' || c == '\r' || c == '\0') return 0;
    /* UTF-8 box drawing (E2 94 xx) and blocks (E2 96 xx) */
    if (c == 0xe2 && (unsigned char)s[pos+1] == 0x94) return 0; /* border */
    if (c == 0xe2 && (unsigned char)s[pos+1] == 0x96) return 1; /* block fill */
    /* ASCII fills: #, @, %, :, or any other printable */
    return (c >= 0x21 && c <= 0x7e);
}

/* Parse a single frame (12+ lines), return bitmask of visible cells, or -1 on EOF/error */
static int parse_frame(FILE *f) {
    char line[256];
    int cells_visible[8] = {0}; /* indexed by cell_idx */
    int content_row = 0;

    /* Skip until we find a top border line */
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '.' || line[0] == '\xe2' || strchr(line, '-')) break;
        if (feof(f)) return -1;
    }
    if (feof(f)) return -1;

    /* Read 10 content rows */
    while (content_row < 10 && fgets(line, sizeof(line), f)) {
        /* Skip empty lines */
        if (line[0] == '\n' || line[0] == '\r') continue;

        /* Detect border lines by checking for corner chars */
        unsigned char c0 = (unsigned char)line[0];
        unsigned char c1 = (unsigned char)line[1];
        unsigned char c2 = (unsigned char)line[2];

        /* ASCII corners: . or ' at start */
        if (c0 == '.' || c0 == '\'') break;  /* Top/bottom border */

        /* UTF-8 corners: E2 94 8C (┌) or E2 94 94 (└) or E2 94 9C (├) */
        if (c0 == 0xe2 && c1 == 0x94 && (c2 == 0x8c || c2 == 0x94 || c2 == 0x9c)) break;

        /* Skip horizontal divider lines (contain - but no fill chars) */
        if (strchr(line, '-') && !strchr(line, '#') && !strchr(line, '@') &&
            !strchr(line, '%') && !strchr(line, ':') &&
            !strstr(line, "\xe2\x96")) continue;

        /* Parse content: check each of 6 grid columns */
        int pos = 0;
        /* Skip left border (ASCII | or UTF-8 │) */
        if ((unsigned char)line[pos] == 0xe2) pos += 3;
        else if (line[pos] == '|') pos += 1;

        for (int c = 0; c < 6 && line[pos]; c++) {
            /* Check for fill in this 2-char-wide column */
            int filled = is_fill_at(line, pos);

            /* Advance by character width (2 ASCII chars or 2 UTF-8 chars) */
            if ((unsigned char)line[pos] == 0xe2) pos += 6; /* 2 × 3-byte UTF-8 */
            else pos += 2;

            /* Skip potential grid dividers */
            while (line[pos] == '|' || ((unsigned char)line[pos] == 0xe2 &&
                   (unsigned char)line[pos+1] == 0x94)) {
                if ((unsigned char)line[pos] == 0xe2) pos += 3;
                else pos++;
            }

            if (filled && content_row < 10) {
                int cell = GRID[content_row][c];
                cells_visible[cell_idx(cell)] = 1;
            }
        }
        content_row++;
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

        if (valid) {
            /* Reconstruct k from extracted bits */
            uint64_t k3_val = 0;
            for (int i = 0; i < n3; i++) {
                k3_val = k3_val * 3 + k3_digits[i];
            }
            uint64_t k_candidate = k3_val * 1073741824ULL + (k2_bits | k4_bits);

            /* Verify: check that k_candidate reproduces all observed masks */
            int verified = 1;
            for (int s = 0; s < 60 && verified; s++) {
                int frame_idx = (s + rot) % 60;
                int expected_idx = perm_index(k_candidate, s);
                uint8_t expected_mask = COMBOS[s][expected_idx];
                if (expected_mask != masks[frame_idx]) verified = 0;
            }

            if (verified) {
                *out_k = k_candidate;
                return rot;
            }
        }
    }
    return -1; /* Failed */
}

static int run_inverse(void) {
    uint8_t masks[60];
    int count = 0;

    fprintf(stderr, "Reading frames from stdin...\n");

    while (count < 60) {
        int mask = parse_frame(stdin);
        if (mask < 0) break;
        masks[count] = (uint8_t)mask;
        int sum = mask_to_sum(mask);
        fprintf(stderr, "\rFrame %d: sum=%d  ", count + 1, sum);
        count++;
    }
    fprintf(stderr, "\n");

    if (count < 60) {
        fprintf(stderr, "Error: Need 60 frames, got %d\n", count);
        return 1;
    }

    fprintf(stderr, "Reconstructing minute number...\n");
    uint64_t k;
    int rot = reconstruct_k(masks, &k);
    if (rot < 0) {
        fprintf(stderr, "Error: Could not reconstruct k\n");
        return 1;
    }

    printf("Minute (k): %llu\n", (unsigned long long)k);
    printf("Rotation: %d (first frame was second %d)\n", rot, rot);

    /* Compute and display origin timestamp */
    time_t now = time(NULL);
    time_t origin = now - (time_t)(k * 60 + rot);
    struct tm local_tm = {0}, utc_tm = {0};
    struct tm *tmp = localtime(&origin);
    if (tmp) local_tm = *tmp;
    tmp = gmtime(&origin);
    if (tmp) utc_tm = *tmp;
    printf("Origin: %04d-%02d-%02d %02d:%02d:%02d (local) = %04d-%02d-%02dT%02d:%02d:%02dZ\n",
           local_tm.tm_year + 1900, local_tm.tm_mon + 1, local_tm.tm_mday,
           local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec,
           utc_tm.tm_year + 1900, utc_tm.tm_mon + 1, utc_tm.tm_mday,
           utc_tm.tm_hour, utc_tm.tm_min, utc_tm.tm_sec);
    return 0;
}

static void usage(const char *prog) {
    printf("Mondrian Terminal Clock\n\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Modes:\n");
    printf("  (default)   Run clock, display frames (1/sec)\n");
    printf("  -l          Live: in-place update (no scroll, requires TTY)\n");
    printf("  -i          Inverse: read 60 frames from stdin, output k\n");
    printf("  -n N        Output N frames fast (no delay), then exit\n\n");
    printf("Display:\n");
    printf("  -a          ASCII mode (.|'#)\n");
    printf("  -u          Unicode mode (box drawing + blocks) [default]\n");
    printf("  -d          Distinct fills (3-color graph coloring)\n");
    printf("  -f CHARS    Custom fill chars for cells 1,2,4,6,12,15,20 (7 chars)\n\n");
    printf("Time:\n");
    printf("  -o ORIGIN   Custom origin (ISO 8601, e.g. 2000-01-01T00:00:00Z)\n");
    printf("  -k K        Use minute K directly (ignores system time)\n\n");
    printf("Examples:\n");
    printf("  %s                    # Live clock\n", prog);
    printf("  %s -k 12345 -n 60     # Generate 60 frames for minute 12345\n", prog);
    printf("  %s -a -f 1246FPT      # ASCII with custom fills\n", prog);
    printf("  %s -i < frames.txt    # Decode frames, output k\n", prog);
}

int main(int argc, char **argv) {
    int unicode = 1, distinct = 0, inverse = 0, live_inplace = 0;
    time_t origin = 0;  /* Unix epoch default, same as webpage */
    int64_t fixed_k = -1;  /* -1 means use system time */
    int64_t num_frames = -1;  /* -1 means infinite (live mode) */
    const char *fill_chars = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-a") == 0) unicode = 0;
        else if (strcmp(argv[i], "-u") == 0) unicode = 1;
        else if (strcmp(argv[i], "-d") == 0) distinct = 1;
        else if (strcmp(argv[i], "-l") == 0) live_inplace = 1;
        else if (strcmp(argv[i], "-i") == 0) inverse = 1;
        else if (strcmp(argv[i], "-f") == 0 && i+1 < argc)
            fill_chars = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc)
            origin = parse_origin(argv[++i]);
        else if (strcmp(argv[i], "-k") == 0 && i+1 < argc)
            fixed_k = atoll(argv[++i]);
        else if (strcmp(argv[i], "-n") == 0 && i+1 < argc)
            num_frames = atoll(argv[++i]);
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]); return 0;
        }
    }

    if (inverse) return run_inverse();

    if (unicode) { enable_utf8(); set_unicode(distinct); }
    else { set_ascii(distinct, fill_chars); }

    signal(SIGINT, handle_signal);
#ifndef _WIN32
    signal(SIGTERM, handle_signal);
#endif

    uint64_t frame = 0;
    while (running && (num_frames < 0 || frame < (uint64_t)num_frames)) {
        time_t now = time(NULL);
        uint64_t k;
        int sec;

        if (fixed_k >= 0) {
            /* Demo mode: cycle through seconds for fixed k */
            k = (uint64_t)fixed_k;
            sec = (int)(frame % 60);
        } else {
            /* Live mode: use system time relative to origin */
            time_t elapsed = now - origin;
            k = (uint64_t)(elapsed / 60);
            sec = (int)(elapsed % 60);
        }

        int idx = perm_index(k, sec);
        uint8_t mask = COMBOS[sec][idx];

        /* In-place mode: move cursor up to overwrite previous frame */
        if (live_inplace && frame > 0) printf("\033[13A");

        render(mask);

        frame++;
        if (num_frames < 0) SLEEP_MS(1000);  /* Only sleep in live mode */
    }

    /* On exit, print local system time (for computing clock start from decoded k) */
    time_t end_time = time(NULL);
    struct tm local_tm = {0}, utc_tm = {0};
    struct tm *tmp = localtime(&end_time);
    if (tmp) local_tm = *tmp;
    tmp = gmtime(&end_time);
    if (tmp) utc_tm = *tmp;
    fprintf(stderr, "\n--- %04d-%02d-%02d %02d:%02d:%02d (local) = %04d-%02d-%02dT%02d:%02d:%02dZ ---\n",
           local_tm.tm_year + 1900, local_tm.tm_mon + 1, local_tm.tm_mday,
           local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec,
           utc_tm.tm_year + 1900, utc_tm.tm_mon + 1, utc_tm.tm_mday,
           utc_tm.tm_hour, utc_tm.tm_min, utc_tm.tm_sec);

    return 0;
}
