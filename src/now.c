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
    printf("  -i          Inverse: read frames, output elapsed time\n");
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
    printf("  -P PERIOD   Signature period (must be coprime with 60)\n");
    printf("              Valid periods have no factors 2, 3, or 5\n");
    printf("              Examples: 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 49...\n");
    printf("  -N VALUE    Encode VALUE (0 to PERIOD-1) in signature (default: 1)\n\n");
    printf("Examples:\n");
    printf("  %s                        # Live clock (original 88B-year period)\n", prog);
    printf("  %s -l -p emoji            # In-place with emoji\n", prog);
    printf("  %s -n 60 -s | %s -i       # Round-trip test\n", prog, prog);
    printf("  %s -P 7 -N 3 -n 60 -s     # Encode value 3 with period 7\n", prog);
    printf("  %s -P 7 -n 60 -s | %s -P 7 -i  # Round-trip with signature\n", prog, prog);
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

/* Parse a single frame, auto-detecting display mode */
static int parse_frame(FILE *f, uint8_t *mask_out) {
    char line[256];
    int cells_visible[8] = {0};
    int half_width_mode = 0;
    int wide_mode = 0;

    /* Skip until we find a top border line */
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '.' || line[0] == '\xe2' || strchr(line, '-')) break;
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
        unsigned char c0 = (unsigned char)line[0];
        unsigned char c1 = (unsigned char)line[1];
        unsigned char c2 = (unsigned char)line[2];
        if (c0 == '.' || c0 == '\'') break;
        if (c0 == 0xe2 && c1 == 0x94 && (c2 == 0x8c || c2 == 0x94 || c2 == 0x98)) break;
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

/* Verify P by checking ALL available windows give consistent signature */
static int verify_period_full(uint8_t *masks, int num_frames, uint64_t P,
                              uint64_t *out_t, uint64_t *out_sig) {
    clock_params_t test_params;
    clock_params_init(&test_params);
    test_params.sig_period = P;

    int num_windows = num_frames / 60;
    if (num_windows < 1) return 0;

    uint64_t first_t = 0, first_sig = 0;
    int first_rot = -1;

    for (int w = 0; w < num_windows; w++) {
        uint64_t t, sig;
        int rot = inverse_time(masks + w * 60, &test_params, &t, &sig);
        if (rot < 0) return 0;

        if (w == 0) {
            first_t = t;
            first_sig = sig;
            first_rot = rot;
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

static int run_inverse(clock_params_t *params) {
    uint8_t masks[600];  /* Up to 10 minutes */
    int total = 0;
    int align_start = -1;  /* First frame at second 0 */
    int need_aligned = (params->sig_period == 0) ? 120 : 60;  /* 2 min for auto, 1 for known P */

    fprintf(stderr, "Reading frames from stdin...\n");
    while (total < 600) {
        uint8_t m;
        if (parse_frame(stdin, &m) < 0) break;
        masks[total++] = m;
        int sum = mask_to_sum(m) % 60;
        fprintf(stderr, "\rFrame %d: sum=%d  ", total, sum);

        /* Track first aligned position */
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

    if (total < 60) {
        fprintf(stderr, "Error: Need at least 60 frames, got %d\n", total);
        return 1;
    }

    int num_minutes = total / 60;
    fprintf(stderr, "Have %d frames (%d complete minutes)\n", total, num_minutes);

    /* If P specified, just decode */
    if (params->sig_period > 0) {
        uint64_t elapsed_t, sig_value;
        int rot = inverse_time(masks, params, &elapsed_t, &sig_value);
        if (rot < 0) {
            fprintf(stderr, "Error: Could not reconstruct\n");
            return 1;
        }
        int first_sec = (rot == 0) ? 0 : (60 - rot);

        printf("elapsed_seconds: %llu\n", (unsigned long long)elapsed_t);
        printf("minute: %llu\n", (unsigned long long)(elapsed_t / 60));
        printf("first_second: %d\n", first_sec);
        printf("\nsignature:\n");
        printf("  period: %llu\n", (unsigned long long)params->sig_period);
        printf("  N_0: %llu\n", (unsigned long long)params->sig_value);
        /* Compute era from decoded signature value */
        uint64_t N0 = params->sig_value;
        uint64_t P = params->sig_period;
        uint64_t N_era = sig_value;
        uint64_t era = (N_era >= N0) ? (N_era - N0) : (P - N0 + N_era);
        printf("  era: %llu\n", (unsigned long long)era);
        if (era > 0) {
            printf("  N_era: %llu\n", (unsigned long long)N_era);
        }
        return 0;
    }

    /* Fast P detection using k_combined differences between windows.
     * With P=1 (no signature), inverse_time returns raw k_combined.
     * For consecutive windows: delta = k_combined[w+1] - k_combined[w] = P
     */

    uint64_t best_P = 0;
    uint64_t best_t = 0, best_sig = 0;

    if (num_minutes >= 2) {
        /* Find first aligned window (starting at second 0) to avoid slow crossing case */
        int align_offset = 0;
        for (int i = 0; i < 60 && i < total; i++) {
            if (mask_to_sum(masks[i]) % 60 == 0) {
                align_offset = i;
                break;
            }
        }

        int aligned_frames = total - align_offset;
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
            int rot = inverse_time(masks + align_offset + w * 60, &base_params, &t, &sig);
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

            if (consistent && is_coprime_60(delta)) {
                fprintf(stderr, "Detected P=%llu from k_combined differences\n",
                        (unsigned long long)delta);

                /* Verify with full verification (use aligned data) */
                uint64_t test_t, test_sig;
                if (verify_period_full(masks + align_offset, aligned_frames, delta, &test_t, &test_sig)) {
                    best_P = delta;
                    best_t = test_t;
                    best_sig = test_sig;
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
        int first_sec = (int)(best_t % 60);
        printf("elapsed_seconds: %llu\n", (unsigned long long)best_t);
        printf("minute: %llu\n", (unsigned long long)(best_t / 60));
        printf("first_second: %d\n", first_sec);
        if (best_P == 1) {
            printf("\n(no signature encoding detected)\n");
        } else {
            printf("\nauto-detected signature:\n");
            printf("  period: %llu\n", (unsigned long long)best_P);
            printf("  N_era: %llu\n", (unsigned long long)best_sig);
        }
    } else {
        /* No signature detected, try original mode */
        clock_params_t no_sig;
        clock_params_init(&no_sig);
        no_sig.sig_period = 0;

        uint64_t elapsed_t, sig_value;
        int rot = inverse_time(masks, &no_sig, &elapsed_t, &sig_value);
        if (rot >= 0) {
            int first_sec = (rot == 0) ? 0 : (60 - rot);
            printf("elapsed_seconds: %llu\n", (unsigned long long)elapsed_t);
            printf("minute: %llu\n", (unsigned long long)(elapsed_t / 60));
            printf("first_second: %d\n", first_sec);
            printf("\n(no signature detected)\n");
        } else {
            fprintf(stderr, "Error: Could not reconstruct\n");
            return 1;
        }
    }

    return 0;
}

/* ============ Main ============ */

int main(int argc, char **argv) {
    clock_params_t params;
    render_opts_t render;
    clock_params_init(&params);
    render_opts_init(&render);

    int inverse = 0, simulate = 0, inplace = 0;
    int64_t num_frames = -1, start_t = -1;
    time_t origin = 0;
    int n_specified = 0;

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
        else if (strcmp(argv[i], "-p") == 0 && i+1 < argc)
            render_apply_preset(&render, argv[++i]);
        else if (strcmp(argv[i], "-f") == 0 && i+1 < argc)
            render.fills = argv[++i];
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
            params.sig_period = strtoull(argv[++i], NULL, 10);
            if (!is_coprime_60(params.sig_period)) {
                fprintf(stderr, "Warning: Period %llu is not coprime with 60\n",
                        (unsigned long long)params.sig_period);
                fprintf(stderr, "         Signatures may not be independent of second\n");
            }
        }
        else if (strcmp(argv[i], "-N") == 0 && i+1 < argc) {
            params.sig_value = strtoull(argv[++i], NULL, 10);
            n_specified = 1;
        }
    }

    /* Default N₀ to 1 when P is specified but N is not */
    if (params.sig_period > 1 && !n_specified) {
        params.sig_value = 1;
    }

    /* Validate signature value */
    if (params.sig_period > 0 && params.sig_value >= params.sig_period) {
        fprintf(stderr, "Error: Signature value %llu must be < period %llu\n",
                (unsigned long long)params.sig_value,
                (unsigned long long)params.sig_period);
        return 1;
    }

    if (inverse) return run_inverse(&params);

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
        uint8_t mask = get_mask(t, &params);

        if (inplace && frame > 0 && IS_TTY()) printf("\033[13A");

        render_mask(mask, &render, stdout);

        printf("\n");
        fflush(stdout);
        frame++;

        if (!simulate) SLEEP_MS(50);
    }

    /* Output timestamp for inverse */
    time_t end_time = (start_t >= 0) ? origin + elapsed + frame - 1
                                     : (simulate ? now_time + frame - 1 : time(NULL));
    struct tm *utc = gmtime(&end_time);
    printf("%04d-%02d-%02dT%02d:%02d:%02dZ\n",
           utc->tm_year + 1900, utc->tm_mon + 1, utc->tm_mday,
           utc->tm_hour, utc->tm_min, utc->tm_sec);

    return 0;
}
