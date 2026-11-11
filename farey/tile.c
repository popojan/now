// Exact tiling of a W x H grid by one rectangle per weight, minimising the
// sum of |log aspect| (branch and bound).  usage: tile W H w1 w2 ...
// prints the cost, then one "w x y a b" line per cell, or "none".
// TILE_MAX_COST=c in the environment skips tilings costing c or more.
// -n (before W) rejects tilings that a single straight line splits in two;
// -r R makes R:1 (either way round) the ideal cell shape instead of a square.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAXN 16
#define MAXD 64
static int W, H, n, ws[MAXN], nopt[MAXN], oa[MAXN][MAXD], ob[MAXN][MAXD], used[MAXN];
static double oc[MAXN][MAXD], lb[MAXN], best = INFINITY;
static unsigned long long rows[512], full;
static int place[MAXN][4], bestp[MAXN][4], nocut;
static double logr;
static int crossed(int along_x, int at) {  // does some placed cell straddle the line?
    for (int i = 0; i < n; i++) {
        if (!used[i]) continue;
        int lo = along_x ? place[i][0] : place[i][1], len = along_x ? place[i][2] : place[i][3];
        if (lo < at && at < lo + len) return 1;
    }
    return 0;
}
static void rec(double cost, double rest) {
    if (cost + rest >= best - 1e-12) return;
    int y = 0;
    while (y < H && rows[y] == full) y++;
    // rows above y are full and later cells start at >= y, so an uncrossed line y is final
    if (nocut && y > 0 && y < H && !crossed(0, y)) return;
    if (y == H) {
        if (nocut)
            for (int x = 1; x < W; x++) if (!crossed(1, x)) return;
        best = cost;
        for (int i = 0; i < n; i++) for (int j = 0; j < 4; j++) bestp[i][j] = place[i][j];
        return;
    }
    int x = __builtin_ctzll(~rows[y] & full);
    int run = 0;
    while (x + run < W && !(rows[y] >> (x + run) & 1)) run++;
    for (int i = 0; i < n; i++) {
        if (used[i] || (i > 0 && ws[i] == ws[i - 1] && !used[i - 1])) continue;
        for (int o = 0; o < nopt[i]; o++) {
            int a = oa[i][o], b = ob[i][o];
            if (a > run || y + b > H) continue;
            unsigned long long m = ((1ULL << a) - 1) << x;
            int ok = 1;
            for (int r = y; r < y + b && ok; r++) if (rows[r] & m) ok = 0;
            if (!ok) continue;
            for (int r = y; r < y + b; r++) rows[r] |= m;
            used[i] = 1; place[i][0] = x; place[i][1] = y; place[i][2] = a; place[i][3] = b;
            rec(cost + oc[i][o], rest - lb[i]);
            used[i] = 0;
            for (int r = y; r < y + b; r++) rows[r] &= ~m;
        }
    }
}
static int cmp(const void *p, const void *q) { return *(int *)q - *(int *)p; }
int main(int argc, char **argv) {
    for (;;) {
        if (argc > 1 && !strcmp(argv[1], "-n")) { nocut = 1; argv++; argc--; }
        else if (argc > 2 && !strcmp(argv[1], "-r")) { logr = log(atof(argv[2])); argv += 2; argc -= 2; }
        else break;
    }
    if (argc < 4) { fprintf(stderr, "usage: tile [-n] [-r R] W H w1 w2 ...\n"); return 2; }
    W = atoi(argv[1]); H = atoi(argv[2]); n = argc - 3;
    if (W > 63 || H > 511 || n > MAXN) return 2;
    for (int i = 0; i < n; i++) ws[i] = atoi(argv[3 + i]);
    qsort(ws, n, sizeof(int), cmp);
    full = (1ULL << W) - 1;
    double rest = 0;
    for (int i = 0; i < n; i++) {
        lb[i] = INFINITY;
        for (int a = 1; a <= ws[i]; a++) {
            if (ws[i] % a) continue;
            int b = ws[i] / a;
            if (a > W || b > H || nopt[i] == MAXD) continue;
            double c = fabs(fabs(log((double)a / b)) - logr);
            // keep options cheapest-first so good tilings are found early
            int k = nopt[i]++;
            while (k > 0 && oc[i][k - 1] > c) { oa[i][k] = oa[i][k - 1]; ob[i][k] = ob[i][k - 1]; oc[i][k] = oc[i][k - 1]; k--; }
            oa[i][k] = a; ob[i][k] = b; oc[i][k] = c;
            if (c < lb[i]) lb[i] = c;
        }
        if (!nopt[i]) { puts("none"); return 0; }
        rest += lb[i];
    }
    const char *cap = getenv("TILE_MAX_COST");
    if (cap) best = atof(cap);
    rec(0, rest);
    if (!bestp[0][2]) { puts("none"); return 0; }
    printf("%.6f\n", best);
    for (int i = 0; i < n; i++) printf("%d %d %d %d %d\n", ws[i], bestp[i][0], bestp[i][1], bestp[i][2], bestp[i][3]);
}
