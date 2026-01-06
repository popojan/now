/*
 * now.c - Unified CLI for "now" Mondrian clock
 *
 * Default behavior: original now clock (88 billion year period)
 * Extended: custom periods, signatures, Kolmogorov variants
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
    printf("now - Mondrian clock with signatures\n\n");
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
    printf("  -o ORIGIN   Origin timestamp (ISO 8601)\n");
    printf("  -t T        Start at specific second from origin\n\n");
    printf("Extended (infinite):\n");
    printf("  -P PERIOD   Custom period in minutes (default: original 88B-year)\n");
    printf("  -V VARIANTS Kolmogorov variants (extends period by factor)\n");
    printf("  -v VARIANT  Use specific variant (0 to V-1)\n");
    printf("  --sig       Show signature values for all divisors\n\n");
    printf("Examples:\n");
    printf("  %s                    # Live clock (original)\n", prog);
    printf("  %s -l -p emoji        # In-place with emoji\n", prog);
    printf("  %s -P 60 --sig -n 1   # Show signatures for period 60\n", prog);
    printf("  %s -n 60 -s | %s -i   # Round-trip test\n", prog, prog);
}

/* ============ Inverse Mode ============ */

static int is_fill_at(const char *s, int pos) {
    unsigned char c = (unsigned char)s[pos];
    if (c == ' ' || c == '|' || c == '-' || c == '.' || c == '\'' ||
        c == '\n' || c == '\r' || c == '\0') return 0;
    if (c == 0xe2 && (unsigned char)s[pos+1] == 0x94) return 0;
    if (c >= 0x80) return 1;
    return (c >= 0x21 && c <= 0x7e);
}

static int utf8_len(const char *s) {
    unsigned char c = (unsigned char)*s;
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

static int parse_frame(FILE *f, uint8_t *mask_out) {
    char line[256];
    int cells_visible[8] = {0};
    int content_row = 0;

    /* Find top border */
    while (fgets(line, sizeof(line), f)) {
        unsigned char c = (unsigned char)line[0];
        if (c == '.' || (c == 0xe2 && (unsigned char)line[1] == 0x94 && (unsigned char)line[2] == 0x8c))
            break;
        if (feof(f)) return -1;
    }
    if (feof(f)) return -1;

    /* Read content rows */
    while (content_row < 10 && fgets(line, sizeof(line), f)) {
        if (line[0] == '\n' || line[0] == '\r') continue;
        unsigned char c0 = (unsigned char)line[0];
        if (c0 == '.' || c0 == '\'') break;
        if (c0 == 0xe2 && (unsigned char)line[1] == 0x94 &&
            ((unsigned char)line[2] == 0x94 || (unsigned char)line[2] == 0x98)) break;

        int pos = (c0 == 0xe2) ? 3 : ((c0 == '|') ? 1 : 0);

        for (int c = 0; c < 6 && line[pos]; c++) {
            int filled = is_fill_at(line, pos);
            unsigned char ch = (unsigned char)line[pos];
            int advance = (ch >= 0x80) ? utf8_len(line + pos) * 2 : 2;
            if (ch == ' ') advance = 2;
            pos += advance;

            if (filled) {
                int cell = GRID[content_row][c];
                switch(cell) {
                    case 1: cells_visible[1] = 1; break;
                    case 2: cells_visible[2] = 1; break;
                    case 4: cells_visible[3] = 1; break;
                    case 6: cells_visible[4] = 1; break;
                    case 12: cells_visible[5] = 1; break;
                    case 15: cells_visible[6] = 1; break;
                    case 20: cells_visible[7] = 1; break;
                }
            }
        }
        content_row++;
    }

    /* Skip to empty line */
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '\n' || strlen(line) <= 1) break;
    }

    if (content_row < 10) return -1;

    *mask_out = (cells_visible[1] ? 0x01 : 0) |
                (cells_visible[2] ? 0x02 : 0) |
                (cells_visible[3] ? 0x04 : 0) |
                (cells_visible[4] ? 0x08 : 0) |
                (cells_visible[5] ? 0x10 : 0) |
                (cells_visible[6] ? 0x20 : 0) |
                (cells_visible[7] ? 0x40 : 0);
    return 0;
}

static int run_inverse(clock_params_t *params) {
    uint8_t masks[60];
    int total = 0;

    fprintf(stderr, "Reading frames from stdin...\n");

    while (total < 60) {
        uint8_t m;
        if (parse_frame(stdin, &m) < 0) break;
        masks[total++] = m;
        int sum = mask_to_sum(m) % 60;
        fprintf(stderr, "\rFrame %d: sum=%d  ", total, sum);
    }
    fprintf(stderr, "\n");

    if (total < 60) {
        fprintf(stderr, "Error: Need 60 frames, got %d\n", total);
        return 1;
    }

    uint64_t k;
    int rot = inverse_minute(masks, params, &k);
    if (rot < 0) {
        fprintf(stderr, "Error: Could not reconstruct\n");
        return 1;
    }

    int first_sec = (rot == 0) ? 0 : (60 - rot);
    /* k is minute where second 0 appears; if first_sec > 0, first frame is from k-1 */
    uint64_t first_minute = (first_sec > 0 && k > 0) ? k - 1 : k;
    uint64_t elapsed = first_minute * 60 + first_sec;

    printf("elapsed_seconds: %llu\n", (unsigned long long)elapsed);
    printf("minute: %llu\n", (unsigned long long)k);
    printf("first_second: %d\n", first_sec);

    /* Show signatures if using custom period */
    if (params->period != PERIOD_ORIGINAL) {
        uint64_t divisors[64];
        int num_div = get_divisors(params->period, divisors, 64);
        printf("\nsignatures:\n");
        for (int i = 0; i < num_div; i++) {
            uint64_t sig = get_signature(k, divisors[i]);
            printf("  sig[%llu]: %llu\n",
                   (unsigned long long)divisors[i],
                   (unsigned long long)sig);
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

    int inverse = 0, simulate = 0, inplace = 0, show_sig = 0;
    int64_t num_frames = -1, start_t = -1;
    time_t origin = 0;

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
        else if (strcmp(argv[i], "--sig") == 0) show_sig = 1;
        else if (strcmp(argv[i], "-p") == 0 && i+1 < argc)
            render_apply_preset(&render, argv[++i]);
        else if (strcmp(argv[i], "-f") == 0 && i+1 < argc)
            render.fills = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc)
            origin = parse_time(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i+1 < argc)
            start_t = atoll(argv[++i]);
        else if (strcmp(argv[i], "-n") == 0 && i+1 < argc)
            num_frames = atoll(argv[++i]);
        else if (strcmp(argv[i], "-P") == 0 && i+1 < argc)
            params.period = strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "-V") == 0 && i+1 < argc)
            params.num_variants = strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "-v") == 0 && i+1 < argc)
            params.variant = strtoull(argv[++i], NULL, 10);
    }

    if (inverse) return run_inverse(&params);

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

    /* Show signatures header if requested */
    if (show_sig && params.period != PERIOD_ORIGINAL) {
        uint64_t divisors[64];
        int num_div = get_divisors(params.period, divisors, 64);
        printf("# Period: %llu, Divisors: ", (unsigned long long)params.period);
        for (int i = 0; i < num_div; i++) {
            printf("%llu", (unsigned long long)divisors[i]);
            if (i < num_div - 1) printf(", ");
        }
        printf("\n\n");
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

        uint64_t k = (uint64_t)(display_time / 60) % params.period;
        int s = (int)(display_time % 60);

        /* Handle Kolmogorov variants */
        if (params.num_variants > 1) {
            uint64_t total_k = (uint64_t)(display_time / 60);
            params.variant = (total_k / params.period) % params.num_variants;
            k = total_k % params.period;
        }

        uint8_t mask = get_mask(k, s, &params);

        if (inplace && frame > 0 && IS_TTY()) printf("\033[13A");

        render_mask(mask, &render, stdout);

        /* Show signatures if requested */
        if (show_sig && params.period != PERIOD_ORIGINAL) {
            uint64_t divisors[64];
            int num_div = get_divisors(params.period, divisors, 64);
            printf("k=%llu s=%d | ", (unsigned long long)k, s);
            for (int i = 0; i < num_div && divisors[i] <= 20; i++) {
                printf("sig[%llu]=%llu ", (unsigned long long)divisors[i],
                       (unsigned long long)get_signature(k, divisors[i]));
            }
            printf("\n");
        }

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
