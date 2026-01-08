/*
 * now.c - Unified CLI for "now" Mondrian clock
 *
 * Default behavior: original now clock (88 billion year period)
 * Extended: custom signature encoding (period must be coprime with 60)
 *
 * Time model:
 *   t = elapsed seconds since origin
 *   s = t mod 60 (visual second, shown by cell sum)
 *   k = t / 60 (logical minute, determines variant selection)
 *
 * Compile: gcc -O2 -o now now.c core.c render.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#define SLEEP_MS(ms) Sleep(ms)
#define IS_TTY() _isatty(_fileno(stdout))
static void enable_vt(void) {
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD m = 0;
    if (GetConsoleMode(h, &m)) SetConsoleMode(h, m | 0x0004);
    SetConsoleOutputCP(65001);
}
#else
#include <unistd.h>
#define SLEEP_MS(ms) usleep((ms)*1000)
#define IS_TTY() isatty(STDOUT_FILENO)
static void enable_vt(void) {}
#endif

#include "core.h"
#include "core8.h"
#include "render.h"

static volatile int running = 1;

static void handle_signal(int sig) {
    (void)sig;
    running = 0;
}

/* Parse ISO 8601 date */
static time_t parse_time(const char *s) {
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
    return 0;
}

static void usage(const char *prog) {
    printf("now - Mondrian clock with signature encoding\n\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Modes:\n");
    printf("  (default)   Live clock\n");
    printf("  -s          Simulate (fast, no delay)\n");
    printf("  -l          In-place update (TTY only)\n");
    printf("  -i          Inverse: read frames, detect P/N, output origin\n");
    printf("  -e          Error correction (with -i): fix corrupted frames\n");
    printf("  -b          Bits: output cell mask per second (7 bits, LSB=cell 1)\n");
    printf("  -d          Decimal: output cell mask as number (0-127)\n");
    printf("  -r          Raw: output binary byte per second (bit 7 reserved)\n");
    printf("  -8          8-bit mode: use 8 cells (1,2,4,6,12,15,20,30) for 99.8%% entropy\n");
    printf("  -W [a,b,c]  Wave mode: triangle wave 0→90→0 with period 2^a × 3^b × 5^c\n");
    printf("              Default: half split (54,22,12) ~117 bits each for period and message\n");
    printf("  -M VALUE    Message value to encode in wave mode (default: 0)\n");
    printf("  -n N        Output N frames then exit\n\n");
    printf("Display:\n");
    printf("  -a          ASCII borders\n");
    printf("  -p PRESET   Preset: cjk (default), blocks, blocks1, distinct, kanji, emoji\n");
    printf("  -f CHARS    Custom 7 fill characters\n");
    printf("  -1          Half-width mode\n");
    printf("  -w          Wide fills (for CJK/emoji)\n\n");
    printf("Time:\n");
    printf("  -o ORIGIN   Origin timestamp (ISO 8601 or 'now')\n");
    printf("              'now' = start of current minute (synced with wall clock)\n");
    printf("  -t T        Start at specific second (overrides wall clock sync)\n\n");
    printf("Signatures:\n");
    printf("  -P VALUE    Encode VALUE as clock signature (auto-detected from recordings)\n");
    printf("              Must be coprime with 60 (no factors 2, 3, or 5)\n");
    printf("              Examples: 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47...\n");
    printf("  -N SALT     Optional salt for era cycling (default: 0)\n\n");
    printf("Examples:\n");
    printf("  %s                        # Live clock (original 88B-year period)\n", prog);
    printf("  %s -l -p emoji            # In-place with emoji\n", prog);
    printf("  %s -n 60 -s | %s -i       # Round-trip test\n", prog, prog);
    printf("  %s -P 7 -n 60 -s          # Clock with signature 7\n", prog);
    printf("  %s -P 7 -n 120 -s | %s -i # Encode and auto-detect signature\n", prog, prog);
}

/* ============ Inverse Mode ============ */

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

/* UTF-8 helper: get byte length of character */
static int utf8_len(const char *s) {
    unsigned char c = (unsigned char)*s;
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
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

/* Count Unicode characters in content area (between borders) */
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
        p += utf8_len(p);
        count++;
    }
    return count;
}

/* Map cell value to index: 1->1, 2->2, 4->3, 6->4, 12->5, 15->6, 20->7 */
static int cell_idx(int cell) {
    switch(cell) {
        case 1: return 1; case 2: return 2; case 4: return 3;
        case 6: return 4; case 12: return 5; case 15: return 6; case 20: return 7;
    }
    return 0;
}

/* Parse a bits line (7 binary digits) */
static int parse_bits_line(FILE *f, uint8_t *mask_out) {
    char line[32];
    if (!fgets(line, sizeof(line), f)) return -1;
    size_t len = strlen(line);

    /* Support both 7-bit and 8-bit format */
    int nbits = 0;
    if (len >= 8 && line[7] >= '0' && line[7] <= '1') nbits = 8;
    else if (len >= 7) nbits = 7;
    else return -1;

    /* Verify format: binary digits */
    for (int i = 0; i < nbits; i++) {
        if (line[i] != '0' && line[i] != '1') return -1;
    }

    if (nbits == 8) {
        /* 8-bit: positions are 30,20,15,12,6,4,2,1 (MSB=cell 30 on left) */
        *mask_out = ((line[0] == '1') ? 0x80 : 0) |  /* cell 30 */
                    ((line[1] == '1') ? 0x40 : 0) |  /* cell 20 */
                    ((line[2] == '1') ? 0x20 : 0) |  /* cell 15 */
                    ((line[3] == '1') ? 0x10 : 0) |  /* cell 12 */
                    ((line[4] == '1') ? 0x08 : 0) |  /* cell 6 */
                    ((line[5] == '1') ? 0x04 : 0) |  /* cell 4 */
                    ((line[6] == '1') ? 0x02 : 0) |  /* cell 2 */
                    ((line[7] == '1') ? 0x01 : 0);   /* cell 1 */
    } else {
        /* 7-bit: positions are 20,15,12,6,4,2,1 (MSB=cell 20 on left) */
        *mask_out = ((line[0] == '1') ? 0x40 : 0) |  /* cell 20 */
                    ((line[1] == '1') ? 0x20 : 0) |  /* cell 15 */
                    ((line[2] == '1') ? 0x10 : 0) |  /* cell 12 */
                    ((line[3] == '1') ? 0x08 : 0) |  /* cell 6 */
                    ((line[4] == '1') ? 0x04 : 0) |  /* cell 4 */
                    ((line[5] == '1') ? 0x02 : 0) |  /* cell 2 */
                    ((line[6] == '1') ? 0x01 : 0);   /* cell 1 */
    }
    return 0;
}

/* Parse a decimal line (number 0-127) */
static int parse_decimal_line(FILE *f, uint8_t *mask_out) {
    char line[32];
    if (!fgets(line, sizeof(line), f)) return -1;

    /* Skip leading whitespace */
    char *p = line;
    while (*p == ' ') p++;

    /* Verify format: digits only */
    int len = 0;
    while (p[len] >= '0' && p[len] <= '9') len++;
    if (len == 0 || len > 3) return -1;

    int val = atoi(p);
    if (val < 0 || val > 255) return -1;  /* 8-bit mode uses 0-255 */

    *mask_out = (uint8_t)val;
    return 0;
}

/* Parse a raw byte (0-127) */
static int parse_raw_byte(FILE *f, uint8_t *mask_out) {
    int c = getc(f);
    if (c == EOF) return -1;
    /* 8-bit mode uses all 256 values */
    *mask_out = (uint8_t)c;
    return 0;
}

/* Parse a single frame, auto-detecting display mode */
static int parse_frame(FILE *f, uint8_t *mask_out) {
    char line[256];
    int cells_visible[8] = {0};
    int half_width_mode = 0;
    int wide_mode = 0;

    /* Skip until we find a top border line (top-left corner) */
    while (fgets(line, sizeof(line), f)) {
        if (is_top_left_corner(line)) break;
        if (feof(f)) return -1;
    }
    if (feof(f)) return -1;

    /* Detect half-width mode from border: 6 dashes = half, 12 = normal */
    int dashes = count_border_dashes(line);
    half_width_mode = (dashes <= 8);

    /* Store content lines for mode detection and parsing */
    char content_lines[10][256];
    int num_content_lines = 0;

    while (num_content_lines < 10 && fgets(line, sizeof(line), f)) {
        if (line[0] == '\n' || line[0] == '\r') continue;
        /* Stop at bottom border or next frame's top border */
        if (is_bottom_border_start(line) || is_top_left_corner(line)) break;
        /* Skip horizontal separator lines (contain dashes but no fill chars) */
        if (strchr(line, '-') && !strchr(line, '#') && !strchr(line, '@') &&
            !strchr(line, '%') && !strstr(line, "\xe2\x96")) continue;
        strncpy(content_lines[num_content_lines], line, 255);
        content_lines[num_content_lines][255] = '\0';
        num_content_lines++;
    }

    /* Detect wide mode vs doubled mode using character count.
     * In doubled mode, fills are printed twice, run lengths always even.
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
            /* Check for odd-length runs of identical characters */
            int found_odd_run = 0;
            for (int i = 0; i < num_content_lines && !found_odd_run; i++) {
                const char *p = content_lines[i];
                if ((unsigned char)*p == 0xe2) p += 3;
                else if (*p == '|') p += 1;
                while (*p && *p != '\n' && !found_odd_run) {
                    unsigned char c = (unsigned char)*p;
                    if (c == ' ' || c == '|') { p++; continue; }
                    if (c == 0xe2 && (unsigned char)*(p+1) == 0x94) { p += 3; continue; }
                    int char_len = utf8_len(p);
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

    /* Parse content rows with detected mode */
    for (int row = 0; row < num_content_lines; row++) {
        char *cline = content_lines[row];
        int pos = 0;
        /* Skip left border */
        if ((unsigned char)cline[pos] == 0xe2) pos += 3;
        else if (cline[pos] == '|') pos += 1;

        for (int c = 0; c < 6 && cline[pos]; c++) {
            int filled = is_fill_at(cline, pos);
            unsigned char ch = (unsigned char)cline[pos];
            int advance;

            if (half_width_mode) {
                /* Half-width: 1 column per cell */
                advance = (ch >= 0x80) ? utf8_len(cline + pos) : 1;
            } else if (wide_mode) {
                /* Wide: 1 fill char per cell, but 2 spaces for empty */
                if (ch == ' ') advance = 2;
                else advance = (ch >= 0x80) ? utf8_len(cline + pos) : 1;
            } else {
                /* Normal: 2 characters per cell (doubled fills) */
                if (ch >= 0x80) {
                    advance = utf8_len(cline + pos) * 2;
                } else {
                    advance = 2;
                }
            }
            pos += advance;

            if (filled) {
                int cell = GRID[row][c];
                cells_visible[cell_idx(cell)] = 1;
            }
        }
    }

    /* Skip to empty line */
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '\n' || strlen(line) <= 1) break;
    }

    if (num_content_lines < 10) return -1;

    *mask_out = (cells_visible[1] ? 0x01 : 0) |
                (cells_visible[2] ? 0x02 : 0) |
                (cells_visible[3] ? 0x04 : 0) |
                (cells_visible[4] ? 0x08 : 0) |
                (cells_visible[5] ? 0x10 : 0) |
                (cells_visible[6] ? 0x20 : 0) |
                (cells_visible[7] ? 0x40 : 0);
    return 0;
}

/* Verify P by checking ALL available windows give consistent signature. */
static int verify_period_full(uint8_t *masks, int num_frames, uint64_t P,
                              int64_t N0, uint64_t *out_t, uint64_t *out_sig) {
    clock_params_t test_params;
    clock_params_init(&test_params);
    test_params.sig_period = P;
    test_params.sig_value = (uint64_t)N0;

    int num_windows = num_frames / 60;
    if (num_windows < 1) return 0;

    uint64_t first_t = 0, first_sig = 0;

    for (int w = 0; w < num_windows; w++) {
        uint64_t t, sig;
        int rot = inverse_time(masks + w * 60, &test_params, &t, &sig);
        if (rot < 0) return 0;

        if (w == 0) {
            first_t = t;
            first_sig = sig;
        } else {
            /* Signature must be identical across all windows */
            if (sig != first_sig) return 0;
            /* Time should increase by ~60 per window */
            int64_t expected_t = (int64_t)first_t + w * 60;
            if ((int64_t)t < expected_t - 5 || (int64_t)t > expected_t + 5) return 0;
        }
    }

    *out_t = first_t;
    *out_sig = first_sig;
    return 1;
}

/*
 * Inverse modes:
 *   1. No -N: Assume era=0, N_0 = decoded N_era (simple origin recovery)
 *   2. -N given: Use explicit N_0, calculate era from decoded N_era
 *
 * Origin computation:
 *   - Simulated (-s): origin = now - elapsed_t (frames are instant)
 *   - Live: origin = start_time - elapsed_t (accounts for real-time delay)
 */
static int run_inverse(clock_params_t *params, clock8_params_t *params8, int n_specified, int simulate, int error_correct, int forced_format, int mode8, int wave_mode, const wave_period_t *wave_period) {
    uint8_t masks[600];  /* Up to 10 minutes */
    frame_correction_t corrections[600];
    int total = 0;
    int align_start = -1;  /* First frame at second 0 */
    /* Wave mode needs 180 frames; P auto-detection needs 2 aligned minutes */
    int p_specified = mode8 ? !U128_EQ(params8->sig_period, U128_ZERO) : (params->sig_period != 0);
    int need_aligned = wave_mode ? WAVE_PERIOD : (p_specified ? 60 : 120);
    time_t start_time = 0;  /* Set when first frame is received */
    int input_format = forced_format;  /* -1 = auto, 0 = visual, 1 = bits, 2 = decimal, 3 = raw */

    if (mode8) {
        fprintf(stderr, "8-bit mode: expecting decimal, bits, or raw format\n");
    }
    fprintf(stderr, "Reading frames from stdin...\n");
    while (total < 600) {
        uint8_t m;
        int ret;

        /* Auto-detect format from first few chars */
        if (input_format < 0) {
            int c = getc(stdin);
            if (c == EOF) break;
            int c2 = getc(stdin);
            if (c2 != EOF) ungetc(c2, stdin);
            ungetc(c, stdin);

            /* Raw: second byte is not newline and first byte is 0-127 */
            if (c <= 127 && c2 != '\n' && c2 != '\r' && c2 != EOF &&
                !(c >= '0' && c <= '9') && c != '|' && c != '-') {
                input_format = 3;  /* raw binary */
            } else if ((c == '0' || c == '1') && (c2 == '0' || c2 == '1')) {
                input_format = 1;  /* bits: starts with two binary digits */
            } else if (c >= '0' && c <= '9') {
                input_format = 2;  /* decimal: starts with digit */
            } else {
                input_format = 0;  /* visual */
            }
            const char *fmt_name[] = {"visual", "bits", "decimal", "raw"};
            fprintf(stderr, "Detected %s format\n", fmt_name[input_format]);
        }

        if (input_format == 1) ret = parse_bits_line(stdin, &m);
        else if (input_format == 2) ret = parse_decimal_line(stdin, &m);
        else if (input_format == 3) ret = parse_raw_byte(stdin, &m);
        else ret = parse_frame(stdin, &m);
        if (ret < 0) {
            if (feof(stdin)) break;  /* End of input */
            continue;  /* Discard truncated frame, try next */
        }
        /* Record time when first frame arrives */
        if (total == 0) start_time = time(NULL);
        masks[total++] = m;
        /* Wave mode uses full sum (0-90), others use mod 60 */
        int sum = wave_mode ? mask8_to_full_sum(m) :
                  (mode8 ? mask8_to_sum(m) : (mask_to_sum(m) % 60));
        fprintf(stderr, "\rFrame %d: sum=%d  ", total, sum);

        /* Track first aligned position (sum=0) */
        if (align_start < 0 && sum == 0) {
            align_start = total - 1;
        }

        /* Stop early if we have enough aligned data */
        if (align_start >= 0 && (total - align_start) >= need_aligned) {
            fprintf(stderr, "\rHave enough data, stopping read.      \n");
            break;
        }
    }
    fprintf(stderr, "\n");

    /* Close stdin to signal encoder to stop (triggers SIGPIPE) */
    fclose(stdin);

    if (total < 60) {
        fprintf(stderr, "Error: Need at least 60 frames, got %d\n", total);
        return 1;
    }

    int num_minutes = total / 60;
    fprintf(stderr, "Have %d frames (%d complete minutes)\n", total, num_minutes);

    /* Error correction if requested */
    if (error_correct) {
        int anchor_count = 0;
        int num_corrected = correct_errors(masks, total, corrections, &anchor_count);

        if (num_corrected < 0) {
            fprintf(stderr, "Error correction failed: <50%% consensus\n");
        } else {
            fprintf(stderr, "Error correction: %d anchors, %d corrected\n",
                    anchor_count, num_corrected);

            if (num_corrected > 0) {
                fprintf(stderr, "Corrections:\n");
                for (int i = 0; i < total; i++) {
                    if (corrections[i].distance > 0) {
                        fprintf(stderr, "  Frame %d: sum %d->%d, mask 0x%02x->0x%02x (dist=%d)\n",
                                i,
                                corrections[i].received_sum,
                                corrections[i].expected_sum,
                                corrections[i].original,
                                corrections[i].corrected,
                                corrections[i].distance);
                    }
                }
            }
        }
    }

    uint64_t elapsed_t = 0;
    uint64_t sig_P = 0, sig_N0 = 0, sig_Nera = 0;

    /* Wave mode inverse: need 180 frames for one cycle */
    if (wave_mode) {
        if (total < WAVE_PERIOD) {
            fprintf(stderr, "Error: Wave mode needs %d frames, got %d\n", WAVE_PERIOD, total);
            return 1;
        }

        /* Find alignment: first frame with sum 0 */
        int wave_align = -1;
        for (int i = 0; i <= total - WAVE_PERIOD; i++) {
            int sum = mask8_to_full_sum(masks[i]);
            if (sum == 0) {
                wave_align = i;
                break;
            }
        }

        if (wave_align < 0) {
            fprintf(stderr, "Error: Could not find wave alignment (sum=0 frame)\n");
            return 1;
        }

        fprintf(stderr, "Wave alignment at frame %d\n", wave_align);

        uint128_t out_cycle, out_msg;
        int ret = inverse_wave(masks + wave_align, wave_period, &out_cycle, &out_msg);
        if (ret < 0) {
            fprintf(stderr, "Error: inverse_wave failed\n");
            return 1;
        }

        /* Compute elapsed time at first input frame:
         * Aligned frame is at elapsed = cycle * WAVE_PERIOD
         * First input frame is wave_align frames before that */
        uint128_t align_elapsed = U128_MUL(out_cycle, WAVE_PERIOD);
        uint128_t total_frames = U128_SUB(align_elapsed, U128_FROM_U64(wave_align));

        char cycle_str[50], msg_str[50], frames_str[50];
        u128_to_str(out_cycle, cycle_str, sizeof(cycle_str));
        u128_to_str(out_msg, msg_str, sizeof(msg_str));
        u128_to_str(total_frames, frames_str, sizeof(frames_str));

        printf("cycle: %s\n", cycle_str);
        printf("message: %s\n", msg_str);
        printf("total_frames: %s\n", frames_str);

        /* Compute origin using start_time (when decoder began reading) */
        uint64_t t64 = U128_LO(total_frames);
        time_t origin_time = start_time - (time_t)t64;
        struct tm *utc = gmtime(&origin_time);
        if (utc) {
            printf("\norigin: %04d-%02d-%02dT%02d:%02d:%02dZ\n",
                   utc->tm_year + 1900, utc->tm_mon + 1, utc->tm_mday,
                   utc->tm_hour, utc->tm_min, utc->tm_sec);
        }

        return 0;
    }

    /* Find alignment offset (first frame at second 0) */
    int global_align_offset = 0;
    for (int i = 0; i < 60 && i < total; i++) {
        int sum = mode8 ? mask8_to_sum(masks[i]) : (mask_to_sum(masks[i]) % 60);
        if (sum == 0) {
            global_align_offset = i;
            break;
        }
    }

    /* 8-bit mode: use inverse8_time with full 128-bit support */
    if (mode8) {
        int aligned_frames = total - global_align_offset;
        int aligned_minutes = aligned_frames / 60;

        /* P auto-detection: need at least 2 aligned minutes */
        uint128_t detected_P = U128_ZERO;
        uint128_t one = U128_FROM_U64(1);

        if (U128_EQ(params8->sig_period, U128_ZERO) && aligned_minutes >= 2) {
            /* Reconstruct k_combined for consecutive minutes */
            uint128_t k0_t, k0_sig, k1_t, k1_sig;
            clock8_params_t raw_params8;
            clock8_params_init(&raw_params8);
            raw_params8.sig_period = one;  /* P=1 means raw k_combined */

            int rot0 = inverse8_time(masks + global_align_offset, &raw_params8, &k0_t, &k0_sig);
            int rot1 = inverse8_time(masks + global_align_offset + 60, &raw_params8, &k1_t, &k1_sig);

            if (rot0 >= 0 && rot1 >= 0 && U128_LT(k0_t, k1_t)) {
                uint128_t k0 = U128_DIV(k0_t, 60);
                uint128_t k1 = U128_DIV(k1_t, 60);
                uint128_t delta = U128_SUB(k1, k0);
                if (!U128_EQ(delta, U128_ZERO)) {
                    detected_P = delta;
                    char delta_str[50];
                    u128_to_str(delta, delta_str, sizeof(delta_str));
                    fprintf(stderr, "Auto-detected P=%s from k_combined delta\n", delta_str);
                }
            }
        }

        /* Use detected or specified P */
        clock8_params_t use_params8;
        clock8_params_init(&use_params8);
        if (U128_LT(U128_ZERO, params8->sig_period)) {
            use_params8.sig_period = params8->sig_period;
        } else if (U128_LT(U128_ZERO, detected_P)) {
            use_params8.sig_period = detected_P;
        }

        uint128_t elapsed_t, sig;
        int rot = inverse8_time(masks + global_align_offset, &use_params8, &elapsed_t, &sig);
        if (rot < 0) {
            fprintf(stderr, "Error: Could not reconstruct 8-bit time\n");
            return 1;
        }

        /* Warn if elapsed time is huge (suggests P is needed but wasn't detected) */
        if (U128_EQ(use_params8.sig_period, U128_ZERO) && U128_HI(elapsed_t) > 0) {
            fprintf(stderr, "Warning: elapsed time > 2^64, P probably needed\n");
            fprintf(stderr, "  (need 2 aligned minutes for auto-detection, have %d)\n",
                    aligned_minutes);
        }

        /* Adjust for alignment offset - elapsed_t is from aligned frames,
         * subtract global_align_offset to get actual elapsed time */
        elapsed_t = U128_SUB(elapsed_t, U128_FROM_U64(global_align_offset));
        int first_sec = (60 - global_align_offset) % 60;

        /* Convert 128-bit elapsed_t to time_t (truncate to 64-bit for practical use) */
        uint64_t elapsed_t64 = U128_LO(elapsed_t);
        time_t origin_time = start_time - (time_t)elapsed_t64;

        struct tm *utc = gmtime(&origin_time);
        char origin_str[64];
        if (utc) {
            strftime(origin_str, sizeof(origin_str), "%Y-%m-%dT%H:%M:%SZ", utc);
        } else {
            snprintf(origin_str, sizeof(origin_str), "(out of range)");
        }

        /* Format 128-bit values as strings */
        char elapsed_str[50], minute_str[50], p_str[50], n_str[50];
        u128_to_str(elapsed_t, elapsed_str, sizeof(elapsed_str));
        u128_to_str(U128_DIV(elapsed_t, 60), minute_str, sizeof(minute_str));

        /* Output in same format as 7-bit mode */
        printf("elapsed_seconds: %s\n", elapsed_str);
        printf("minute: %s\n", minute_str);
        printf("first_second: %d\n", first_sec);
        printf("\n");
        printf("origin: %s\n", origin_str);

        if (U128_LT(one, use_params8.sig_period)) {
            u128_to_str(use_params8.sig_period, p_str, sizeof(p_str));
            u128_to_str(sig, n_str, sizeof(n_str));
            printf("\n");
            printf("signature:\n");
            printf("  P: %s\n", p_str);
            printf("  N: %s\n", n_str);
        }

        return 0;
    }

    /* If P specified, decode N_era and go to output */
    if (params->sig_period > 0) {
        /* Use aligned frames for reconstruction */
        clock_params_t raw_params;
        clock_params_init(&raw_params);
        raw_params.sig_period = 1;
        uint64_t raw_t, dummy;
        int rot = inverse_time(masks + global_align_offset, &raw_params, &raw_t, &dummy);
        if (rot < 0) {
            fprintf(stderr, "Error: Could not reconstruct\n");
            return 1;
        }
        uint64_t k_combined = raw_t / 60;
        sig_P = params->sig_period;
        sig_Nera = k_combined % sig_P;
        goto output;
    }

    /* Fast P detection using k_combined differences between windows.
     * With P=1 (no signature), inverse_time returns raw k_combined.
     * For consecutive windows: delta = k_combined[w+1] - k_combined[w] = P
     */

    uint64_t best_P = 0, best_sig = 0;

    if (num_minutes >= 2) {
        int aligned_frames = total - global_align_offset;
        int aligned_minutes = aligned_frames / 60;

        if (aligned_minutes < 2) {
            fprintf(stderr, "Not enough aligned data for delta detection\n");
        }

        /* Reconstruct k_combined for each aligned window with P=1 */
        clock_params_t base_params;
        clock_params_init(&base_params);
        base_params.sig_period = 1;  /* Raw k_combined */

        uint64_t k_combined[10];
        int valid = (aligned_minutes >= 2);

        for (int w = 0; w < aligned_minutes && w < 10 && valid; w++) {
            uint64_t t, sig;
            int rot = inverse_time(masks + global_align_offset + w * 60, &base_params, &t, &sig);
            if (rot < 0) { valid = 0; break; }
            k_combined[w] = t / 60;  /* t = k * 60 when P=1, so k = t/60 */
        }
        num_minutes = aligned_minutes;

        if (valid && num_minutes >= 2 && k_combined[1] > k_combined[0]) {
            /* Compute delta between consecutive windows */
            uint64_t delta = k_combined[1] - k_combined[0];

            /* Verify delta is consistent across all windows */
            int consistent = 1;
            for (int w = 1; w < num_minutes - 1 && consistent; w++) {
                if (k_combined[w + 1] - k_combined[w] != delta)
                    consistent = 0;
            }

            if (consistent && delta > 0) {
                fprintf(stderr, "Detected P=%llu from k_combined differences",
                        (unsigned long long)delta);
                if (!is_coprime_60(delta)) {
                    fprintf(stderr, " (warning: not coprime with 60)");
                }
                fprintf(stderr, "\n");

                /* Verify with full verification, using N_0 (default 0) */
                uint64_t test_t, test_sig;
                if (verify_period_full(masks + global_align_offset, aligned_frames, delta,
                                       (int64_t)params->sig_value, &test_t, &test_sig)) {
                    best_P = delta;
                    best_sig = test_sig;
                    (void)test_t;  /* unused, t computed later from aligned frame */
                }
            }
        }

        if (best_P == 0) {
            fprintf(stderr, "Delta method failed - P may not be consistent\n");
        }
    } else {
        fprintf(stderr, "Cannot auto-detect P with only 1 minute\n");
    }


    if (best_P > 0) {
        sig_P = best_P;
        sig_Nera = best_sig;
    } else {
        /* No signature detected, try original mode */
        clock_params_t no_sig;
        clock_params_init(&no_sig);
        no_sig.sig_period = 0;

        uint64_t sig_value;
        int rot = inverse_time(masks, &no_sig, &elapsed_t, &sig_value);
        if (rot < 0) {
            fprintf(stderr, "Error: Could not reconstruct\n");
            return 1;
        }
        goto output_no_sig;
    }

output:;
    /*
     * We have sig_P and sig_Nera. Now interpret based on mode:
     *   Mode 1 (no -N, no -o): Assume era=0, N_0 = N_era
     *   Mode 2 (-o given):     Verify era=0 against origin, N_0 = N_era
     *   Mode 3 (-N given):     Use explicit N_0, calculate era
     */
    {
        /* Get k_combined by calling inverse_time with P=1
         * Use aligned frames (starting at second 0) to avoid minute-crossing ambiguity */
        clock_params_t raw_params;
        clock_params_init(&raw_params);
        raw_params.sig_period = 1;
        uint64_t raw_t, dummy;
        int rot = inverse_time(masks + global_align_offset, &raw_params, &raw_t, &dummy);
        if (rot < 0) {
            fprintf(stderr, "Error: Could not get k_combined\n");
            return 1;
        }
        uint64_t k_combined = raw_t / 60;

        /* Compute local_minute from aligned frame (within era) */
        uint64_t local_minute = k_combined / sig_P;

        /* first_sec is the second of frame 0 (before alignment) */
        int first_sec = (60 - global_align_offset) % 60;
        uint64_t era = 0;

        if (n_specified) {
            /* Mode 3: Use explicit N_0, calculate era */
            sig_N0 = params->sig_value;
            era = (sig_Nera >= sig_N0) ? (sig_Nera - sig_N0) : (sig_P - sig_N0 + sig_Nera);
            uint64_t max_minute_per_era = PERIOD_ORIGINAL_MINUTES / sig_P;
            elapsed_t = (era * max_minute_per_era + local_minute) * 60 - global_align_offset;
        } else {
            /* Mode 1/2: Assume era=0, N_0 = N_era */
            sig_N0 = sig_Nera;
            elapsed_t = local_minute * 60 - global_align_offset;
        }

        printf("elapsed_seconds: %llu\n", (unsigned long long)elapsed_t);
        printf("minute: %llu\n", (unsigned long long)(elapsed_t / 60));
        printf("first_second: %d\n", first_sec);

        /* Compute origin: use start_time for live mode, now for simulated */
        time_t ref_time = simulate ? time(NULL) : start_time;
        time_t origin = ref_time - (time_t)elapsed_t;
        struct tm *utc = gmtime(&origin);
        if (utc) {
            printf("\norigin: %04d-%02d-%02dT%02d:%02d:%02dZ\n",
                   utc->tm_year + 1900, utc->tm_mon + 1, utc->tm_mday,
                   utc->tm_hour, utc->tm_min, utc->tm_sec);
        }

        printf("\nsignature:\n");
        printf("  P: %llu\n", (unsigned long long)sig_P);
        printf("  N: %llu\n", (unsigned long long)sig_N0);
        if (era > 0) {
            printf("  era: %llu\n", (unsigned long long)era);
            printf("  N_era: %llu\n", (unsigned long long)sig_Nera);
        }
    }
    return 0;

output_no_sig:;
    {
        int first_sec = (int)(elapsed_t % 60);
        printf("elapsed_seconds: %llu\n", (unsigned long long)elapsed_t);
        printf("minute: %llu\n", (unsigned long long)(elapsed_t / 60));
        printf("first_second: %d\n", first_sec);

        /* Compute origin: use start_time for live mode, now for simulated */
        time_t ref_time = simulate ? time(NULL) : start_time;
        time_t origin = ref_time - (time_t)elapsed_t;
        struct tm *utc = gmtime(&origin);
        if (utc) {
            printf("\norigin: %04d-%02d-%02dT%02d:%02d:%02dZ\n",
                   utc->tm_year + 1900, utc->tm_mon + 1, utc->tm_mday,
                   utc->tm_hour, utc->tm_min, utc->tm_sec);
        }

        printf("\n(no signature detected)\n");
    }
    return 0;
}

/* ============ Main ============ */

int main(int argc, char **argv) {
    clock_params_t params;
    render_opts_t render;
    clock_params_init(&params);
    render_opts_init(&render);

    int inverse = 0, simulate = 0, inplace = 0, bits_mode = 0, decimal_mode = 0, raw_mode = 0, error_correct = 0, mode8 = 0;
    int wave_mode = 0;
    wave_period_t wave_period = {0, 0, 0};
    uint128_t wave_msg = U128_ZERO;
    int64_t num_frames = -1, start_t = -1;
    time_t origin = 0;
    int n_specified = 0;  /* Track if -N was explicitly provided */
    const char *p_str = NULL, *n_str = NULL;  /* Store -P/-N strings for later parsing */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]); return 0;
        }
        else if (strcmp(argv[i], "-a") == 0) render.ascii = 1;
        else if (strcmp(argv[i], "-1") == 0) render.half_width = 1;
        else if (strcmp(argv[i], "-w") == 0) render.wide_fills = 1;
        else if (strcmp(argv[i], "-l") == 0) inplace = 1;
        else if (strcmp(argv[i], "-s") == 0) simulate = 1;
        else if (strcmp(argv[i], "-i") == 0) inverse = 1;
        else if (strcmp(argv[i], "-e") == 0) error_correct = 1;
        else if (strcmp(argv[i], "-b") == 0) bits_mode = 1;
        else if (strcmp(argv[i], "-d") == 0) decimal_mode = 1;
        else if (strcmp(argv[i], "-r") == 0) raw_mode = 1;
        else if (strcmp(argv[i], "-8") == 0) mode8 = 1;
        else if (strcmp(argv[i], "-W") == 0) {
            wave_mode = 1;
            mode8 = 1;  /* Wave mode uses 8 cells */
            /* Default to half-and-half split (~117 bits each) */
            wave_period.exp2 = WAVE_EXP2_MAX / 2;  /* 54 */
            wave_period.exp3 = WAVE_EXP3_MAX / 2;  /* 22 */
            wave_period.exp5 = WAVE_EXP5_MAX / 2;  /* 12 */
            /* Override if explicit values provided */
            if (i+1 < argc && argv[i+1][0] != '-') {
                int a, b, c;
                if (sscanf(argv[++i], "%d,%d,%d", &a, &b, &c) == 3) {
                    wave_period.exp2 = a;
                    wave_period.exp3 = b;
                    wave_period.exp5 = c;
                } else {
                    fprintf(stderr, "Error: -W format is a,b,c (e.g., -W 0,0,24)\n");
                    return 1;
                }
            }
        }
        else if (strcmp(argv[i], "-M") == 0 && i+1 < argc) {
            wave_msg = str_to_u128(argv[++i]);
        }
        else if (strcmp(argv[i], "-p") == 0 && i+1 < argc)
            render_apply_preset(&render, argv[++i]);
        else if (strcmp(argv[i], "-f") == 0 && i+1 < argc) {
            render.fills = argv[++i];
            render.wide_fills = 0;  /* Custom fills are doubled unless -w */
        }
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) {
            if (strcmp(argv[i+1], "now") == 0) {
                /* Floor to start of current minute */
                origin = (time(NULL) / 60) * 60;
                i++;
            } else {
                origin = parse_time(argv[++i]);
            }
        }
        else if (strcmp(argv[i], "-t") == 0 && i+1 < argc)
            start_t = atoll(argv[++i]);
        else if (strcmp(argv[i], "-n") == 0 && i+1 < argc)
            num_frames = atoll(argv[++i]);
        else if (strcmp(argv[i], "-P") == 0 && i+1 < argc) {
            p_str = argv[++i];
        }
        else if (strcmp(argv[i], "-N") == 0 && i+1 < argc) {
            n_str = argv[++i];
            n_specified = 1;
        }
    }

    /* Parse P and N - use 128-bit for 8-bit mode, 64-bit otherwise */
    clock8_params_t params8;
    clock8_params_init(&params8);

    if (p_str) {
        if (mode8) {
            params8.sig_period = str_to_u128(p_str);
        } else {
            params.sig_period = strtoull(p_str, NULL, 10);
        }
    }
    if (n_str) {
        if (mode8) {
            params8.sig_value = str_to_u128(n_str);
        } else {
            params.sig_value = strtoull(n_str, NULL, 10);
        }
    }


    /* 8-bit mode requires numeric output (visual only shows 7 cells) */
    if (mode8 && !bits_mode && !decimal_mode && !raw_mode && !inverse) {
        fprintf(stderr, "Warning: -8 mode requires -b, -d, or -r for output (visual only shows 7 cells)\n");
        fprintf(stderr, "         Cell 30 will not be displayed. Use -d for decimal output.\n");
    }

    /* Validate signature value */
    if (params.sig_period > 0 && params.sig_value >= params.sig_period) {
        fprintf(stderr, "Error: Signature value %llu must be < period %llu\n",
                (unsigned long long)params.sig_value,
                (unsigned long long)params.sig_period);
        return 1;
    }

    /* Validate P not exceeding clock period */
    if (params.sig_period > PERIOD_ORIGINAL_MINUTES) {
        fprintf(stderr, "Error: P=%llu exceeds clock period (%llu minutes)\n",
                (unsigned long long)params.sig_period,
                (unsigned long long)PERIOD_ORIGINAL_MINUTES);
        return 1;
    }

    /* Warn if era length is short (< 100 years) */
    if (params.sig_period > 0) {
        uint64_t era_minutes = PERIOD_ORIGINAL_MINUTES / params.sig_period;
        double era_years = (double)era_minutes / (60.0 * 24.0 * 365.25);
        if (era_years < 100.0) {
            fprintf(stderr, "Warning: P too large, era cycles every %.1f years.\n", era_years);
            fprintf(stderr, "         Origin recovery requires -N to specify N_0.\n");
        }
    }

    /* 8-bit mode: validate P against period */
    if (mode8 && !U128_EQ(params8.sig_period, U128_ZERO)) {
        uint128_t period8 = str_to_u128(PERIOD8_STR);

        /* Error if P > period */
        if (!U128_LT(params8.sig_period, period8) && !U128_EQ(params8.sig_period, period8)) {
            char p_str_buf[50];
            u128_to_str(params8.sig_period, p_str_buf, sizeof(p_str_buf));
            fprintf(stderr, "Error: P=%s exceeds clock period (%s minutes)\n",
                    p_str_buf, PERIOD8_STR);
            return 1;
        }

        /* Warn if era < 100 years */
        /* era_minutes = period / P; era_years = era_minutes / (60 * 24 * 365.25) */
        uint128_t era_minutes;
        u128_divmod(period8, params8.sig_period, &era_minutes, NULL);
        /* Approximate: if era_minutes < 52596000 (100 years in minutes), warn */
        uint64_t min_100_years = 100ULL * 365 * 24 * 60;  /* ~52.6M */
        if (U128_HI(era_minutes) == 0 && U128_LO(era_minutes) < min_100_years) {
            double era_years = (double)U128_LO(era_minutes) / (60.0 * 24.0 * 365.25);
            fprintf(stderr, "Warning: P too large, era cycles every %.1f years.\n", era_years);
            fprintf(stderr, "         Origin recovery requires -N to specify N_0.\n");
        }
    }

    /* Validate wave mode parameters */
    if (wave_mode) {
        /* Warn if -P/-N specified with wave mode */
        if (p_str || n_str) {
            fprintf(stderr, "Warning: -P/-N ignored in wave mode (use -M for message encoding)\n");
        }
        /* Validate combo tables at startup */
        if (validate_wave_combos() < 0) {
            fprintf(stderr, "Error: WAVE_COMBOS table has invalid entries\n");
            return 1;
        }
        if (wave_period.exp2 < 0 || wave_period.exp2 > WAVE_EXP2_MAX) {
            fprintf(stderr, "Error: exp2 must be 0-%d (got %d)\n", WAVE_EXP2_MAX, wave_period.exp2);
            return 1;
        }
        if (wave_period.exp3 < 0 || wave_period.exp3 > WAVE_EXP3_MAX) {
            fprintf(stderr, "Error: exp3 must be 0-%d (got %d)\n", WAVE_EXP3_MAX, wave_period.exp3);
            return 1;
        }
        if (wave_period.exp5 < 0 || wave_period.exp5 > WAVE_EXP5_MAX) {
            fprintf(stderr, "Error: exp5 must be 0-%d (got %d)\n", WAVE_EXP5_MAX, wave_period.exp5);
            return 1;
        }

        /* Validate 128-bit constraint: both period and message must fit in uint128_t
         * Period bits ≈ a + 1.585*b + 2.322*c (using log2(3)≈1.585, log2(5)≈2.322)
         * Message bits ≈ 233.4 - period_bits
         * Constraint: 105 ≤ period_bits ≤ 128 */
        double period_bits = wave_period.exp2 + 1.585 * wave_period.exp3 + 2.322 * wave_period.exp5;
        double msg_bits = 233.4 - period_bits;
        if (period_bits > 128.0) {
            fprintf(stderr, "Error: Period too large (%.1f bits > 128), reduce exponents\n", period_bits);
            return 1;
        }
        if (msg_bits > 128.0) {
            fprintf(stderr, "Error: Message capacity too large (%.1f bits > 128), increase exponents\n", msg_bits);
            fprintf(stderr, "       Minimum: a + 1.585*b + 2.322*c >= 105\n");
            return 1;
        }

        int msg_exp2 = WAVE_EXP2_MAX - wave_period.exp2;
        int msg_exp3 = WAVE_EXP3_MAX - wave_period.exp3;
        int msg_exp5 = WAVE_EXP5_MAX - wave_period.exp5;
        fprintf(stderr, "Wave mode: period 2^%d × 3^%d × 5^%d (~%.0f bits)\n",
                wave_period.exp2, wave_period.exp3, wave_period.exp5, period_bits);
        fprintf(stderr, "           message capacity 2^%d × 3^%d × 5^%d (~%.0f bits)\n",
                msg_exp2, msg_exp3, msg_exp5, msg_bits);
    }

    if (inverse) {
        int forced_format = raw_mode ? 3 : (decimal_mode ? 2 : (bits_mode ? 1 : -1));
        return run_inverse(&params, &params8, n_specified, simulate, error_correct, forced_format, mode8, wave_mode, &wave_period);
    }

    /* Show era info when using signatures */
    if (params.sig_period > 1) {
        time_t now_time = time(NULL);
        int64_t elapsed = (start_t >= 0) ? start_t : (now_time - origin);
        uint64_t minute = (uint64_t)(elapsed / 60);
        uint64_t N0 = params.sig_value;
        uint64_t P = params.sig_period;
        uint64_t max_minute_per_era = (PERIOD_ORIGINAL_MINUTES - N0) / P;
        uint64_t era = minute / max_minute_per_era;

        if (era > 0) {
            uint64_t N_era = (N0 + era) % P;
            double years_per_era = (double)max_minute_per_era / (60.0 * 24 * 365.25);
            fprintf(stderr, "Era %llu: N_era=%llu (era length: %.1e years)\n",
                    (unsigned long long)era, (unsigned long long)N_era, years_per_era);
        }
    }

    enable_vt();
    signal(SIGINT, handle_signal);
#ifndef _WIN32
    signal(SIGTERM, handle_signal);
    signal(SIGPIPE, handle_signal);  /* Stop loop but print timestamp */
#endif

    /* Calculate start time */
    time_t now_time = time(NULL);
    int64_t elapsed;
    if (start_t >= 0) {
        elapsed = start_t;
    } else {
        elapsed = now_time - origin;
    }

    /* Main loop */
    uint64_t frame = 0;
    int64_t next_display = elapsed;

    while (running && (num_frames < 0 || frame < (uint64_t)num_frames)) {
        int64_t display_time;

        if (simulate) {
            display_time = elapsed + (int64_t)frame;
        } else {
            int64_t current = (start_t >= 0) ? elapsed + (time(NULL) - now_time)
                                             : time(NULL) - origin;
            if (current >= next_display) {
                display_time = next_display++;
            } else {
                SLEEP_MS(50);
                continue;
            }
        }

        /* Get mask for this second */
        uint64_t t = (uint64_t)display_time;
        uint8_t mask;

        if (wave_mode) {
            /* Wave mode: triangle wave 0→90→0 encoding */
            uint128_t cycle_idx = U128_FROM_U64(t / WAVE_PERIOD);
            int pos = (int)(t % WAVE_PERIOD);
            mask = get_wave_mask(cycle_idx, wave_msg, pos, &wave_period);
        } else if (mode8) {
            /* 8-bit mode: use 128-bit time and 8 cells */
            uint128_t t128 = U128_FROM_U64(t);
            mask = get_mask8(t128, &params8);
        } else {
            mask = get_mask(t, &params);
        }

        if (bits_mode) {
            /* Output bits: 8 bits in mode8, 7 bits otherwise */
            if (inplace && frame > 0 && IS_TTY()) printf("\r");
            if (mode8) {
                printf("%c%c%c%c%c%c%c%c",
                       (mask & 0x80) ? '1' : '0',  /* cell 30 */
                       (mask & 0x40) ? '1' : '0',  /* cell 20 */
                       (mask & 0x20) ? '1' : '0',  /* cell 15 */
                       (mask & 0x10) ? '1' : '0',  /* cell 12 */
                       (mask & 0x08) ? '1' : '0',  /* cell 6 */
                       (mask & 0x04) ? '1' : '0',  /* cell 4 */
                       (mask & 0x02) ? '1' : '0',  /* cell 2 */
                       (mask & 0x01) ? '1' : '0'); /* cell 1 */
            } else {
                printf("%c%c%c%c%c%c%c",
                       (mask & 0x40) ? '1' : '0',  /* cell 20 */
                       (mask & 0x20) ? '1' : '0',  /* cell 15 */
                       (mask & 0x10) ? '1' : '0',  /* cell 12 */
                       (mask & 0x08) ? '1' : '0',  /* cell 6 */
                       (mask & 0x04) ? '1' : '0',  /* cell 4 */
                       (mask & 0x02) ? '1' : '0',  /* cell 2 */
                       (mask & 0x01) ? '1' : '0'); /* cell 1 */
            }
            if (!inplace || !IS_TTY()) printf("\n");
        } else if (decimal_mode) {
            if (inplace && frame > 0 && IS_TTY()) printf("\r");
            printf("%d", mask);
            if (!inplace || !IS_TTY()) printf("\n");
        } else if (raw_mode) {
            putchar(mask);  /* Binary byte */
        } else {
            if (inplace && frame > 0 && IS_TTY()) printf("\033[13A");
            render_mask(mask, &render, stdout);
            printf("\n");
        }
        if (!simulate) {
            fflush(stdout);
            SLEEP_MS(50);
        }
        frame++;
    }

    return 0;
}
