// usage: farey_search k n S   (gcc -O2 -o farey_search farey_search.c)
// Exhaustive: n sorted weights summing to S whose subset sums contain every S*p/q, q<=k.
// Next weight w <= smallest uncovered required r (all later weights are >= w).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAXS 2600
static int S, n, req[MAXS], nreq; static long nodes, found;
static unsigned char tmp[8][MAXS + 1];
static void dfs(int d, int lo, int rem, unsigned char *m, int *ws) {
    nodes++;
    int slots = n - d;
    if (slots == 1) {
        int w = rem; if (w < lo) return;
        for (int i = 0; i < nreq; i++) { int r = req[i];
            if (!m[r] && !(r >= w && m[r - w])) return; }
        ws[d] = w; found++;
        if (found <= 10) { for (int i = 0; i < n; i++) printf("%d ", ws[i]); printf("\n"); }
        return;
    }
    int r = S;
    for (int i = 0; i < nreq; i++) if (!m[req[i]]) { r = req[i]; break; }
    int hi = rem / slots; if (r < hi) hi = r;
    unsigned char *m2 = tmp[d];
    for (int w = lo; w <= hi; w++) {
        memcpy(m2, m, S + 1);
        for (int s = S - w; s >= 0; s--) if (m[s]) m2[s + w] = 1;
        ws[d] = w; dfs(d + 1, w, rem - w, m2, ws);
    }
}
int main(int argc, char **argv) {
    int k = atoi(argv[1]); n = atoi(argv[2]); S = atoi(argv[3]);
    unsigned char seen[MAXS + 1] = {0};
    for (int q = 1; q <= k; q++) for (int p = 0; p <= q; p++)
        if ((S * p) % q == 0) { int v = S * p / q; if (!seen[v] && v <= S / 2) { seen[v] = 1; } }
        else { printf("S not divisible by %d\n", q); return 1; }
    for (int v = 0; v <= S / 2; v++) if (seen[v]) req[nreq++] = v;  // complements follow by symmetry
    unsigned char m[MAXS + 1] = {0}; m[0] = 1; int ws[16];
    dfs(0, 1, S, m, ws);
    printf("k=%d n=%d S=%d: %ld solutions, %ld nodes\n", k, n, S, found, nodes);
}
