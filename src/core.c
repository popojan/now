/*
 * core.c - Mathematical core implementation for "now" clock
 */

#include "core.h"
#include <string.h>

/* Cell combinations for each second - stored as bitmask
 * Bits: 6=20, 5=15, 4=12, 3=6, 2=4, 1=2, 0=1 */
const uint8_t COMBOS[60][4] = {
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

const uint8_t COMBO_CNT[60] = {
    2,1,1,1,1,1,2,2,1,1,1,1,2,2,1,2,2,2,3,3,3,4,4,3,3,3,3,4,3,2,2,2,3,4,3,3,3,3,4,4,3,3,3,2,2,2,1,2,2,1,1,1,1,2,2,1,1,1,1,1
};

void clock_params_init(clock_params_t *params) {
    params->period = PERIOD_ORIGINAL;
    params->num_variants = 1;
    params->variant = 0;
}

/* Simple hash for variant permutation */
static uint32_t hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;
    return x;
}

/* Original perm_index algorithm (compatible with now.c and web) */
static int perm_index_original(uint64_t k, int s) {
    k = k % PERIOD_ORIGINAL;

    int m = COMBO_CNT[s];
    if (m == 1) return 0;

    /* Count how many 2/3/4-option seconds come before s */
    int n2 = 0, n3 = 0, n4 = 0;
    for (int i = 0; i < s; i++) {
        if (COMBO_CNT[i] == 2) n2++;
        else if (COMBO_CNT[i] == 3) n3++;
        else if (COMBO_CNT[i] == 4) n4++;
    }

    uint64_t k3_val = k / 1073741824ULL;  /* k / 2^30 */
    uint64_t rest = k % 1073741824ULL;    /* k % 2^30 */

    if (m == 2) return (rest >> (29 - n2)) & 1;
    if (m == 3) {
        uint64_t p = 1;
        for (int i = 0; i < 15 - n3; i++) p *= 3;
        return (k3_val / p) % 3;
    }
    if (m == 4) return (rest >> (10 - 2*n4)) & 3;
    return 0;
}

/* Simplified perm_index for custom periods (testing) */
static int perm_index_simple(uint64_t k, int s, uint64_t period) {
    k = k % period;
    int m = COMBO_CNT[s];
    if (m == 1) return 0;
    return (int)((k * 7 + s * 13) % m);
}

int perm_index(uint64_t k, int s, const clock_params_t *params) {
    int base_idx;

    if (params->period == PERIOD_ORIGINAL) {
        base_idx = perm_index_original(k, s);
    } else {
        base_idx = perm_index_simple(k, s, params->period);
    }

    /* Apply Kolmogorov variant permutation if enabled */
    if (params->num_variants > 1 && params->variant > 0) {
        int m = COMBO_CNT[s];
        if (m > 1) {
            uint32_t perm_seed = hash32((uint32_t)(params->variant * 1000 + s));
            base_idx = (base_idx + (int)(perm_seed % m)) % m;
        }
    }

    return base_idx;
}

uint8_t get_mask(uint64_t k, int s, const clock_params_t *params) {
    int idx = perm_index(k, s, params);
    return COMBOS[s][idx];
}

int mask_to_sum(uint8_t mask) {
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

int find_combo_idx(int s, uint8_t mask) {
    for (int i = 0; i < COMBO_CNT[s]; i++) {
        if (COMBOS[s][i] == mask) return i;
    }
    return -1;
}

/* ============ Divisors and Signatures ============ */

int get_divisors(uint64_t n, uint64_t *out, int max_out) {
    if (n == 0 || max_out <= 0) return 0;

    int count = 0;
    for (uint64_t i = 1; i * i <= n && count < max_out; i++) {
        if (n % i == 0) {
            out[count++] = i;
            if (i != n / i && count < max_out) {
                out[count++] = n / i;
            }
        }
    }

    /* Simple bubble sort (divisor count is typically small) */
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (out[i] > out[j]) {
                uint64_t tmp = out[i];
                out[i] = out[j];
                out[j] = tmp;
            }
        }
    }

    return count;
}

int factorize(uint64_t n, uint64_t *primes, int *exps, int max_factors) {
    int count = 0;
    uint64_t d = 2;

    while (d * d <= n && count < max_factors) {
        if (n % d == 0) {
            primes[count] = d;
            exps[count] = 0;
            while (n % d == 0) {
                exps[count]++;
                n /= d;
            }
            count++;
        }
        d++;
    }

    if (n > 1 && count < max_factors) {
        primes[count] = n;
        exps[count] = 1;
        count++;
    }

    return count;
}

uint64_t get_signature(uint64_t elapsed_minutes, uint64_t divisor) {
    return elapsed_minutes % divisor;
}

void get_all_signatures(uint64_t elapsed_minutes, uint64_t period,
                        uint64_t *divisors, uint64_t *sigs, int num_divisors) {
    for (int i = 0; i < num_divisors; i++) {
        sigs[i] = elapsed_minutes % divisors[i];
    }
}

/* ============ Inverse (reconstruction) ============ */

/* Verify k against observations, return match count (0-60) */
static int verify_k(uint64_t k, uint8_t masks[60], int rot,
                    const clock_params_t *params) {
    int matches = 0;
    for (int s = 0; s < 60; s++) {
        int frame_idx = (s + rot) % 60;
        uint8_t expected = get_mask(k, s, params);
        if (expected == masks[frame_idx]) matches++;
    }
    return matches;
}

int inverse_minute(uint8_t masks[60], const clock_params_t *params, uint64_t *out_k) {
    /* For original period, use optimized bit extraction */
    if (params->period == PERIOD_ORIGINAL) {
        for (int rot = 0; rot < 60; rot++) {
            uint64_t k2_bits = 0, k4_bits = 0;
            int k3_digits[16];
            int n2 = 0, n3 = 0, n4 = 0;
            int valid = 1;

            for (int s = 0; s < 60 && valid; s++) {
                int frame_idx = (s + rot) % 60;
                uint8_t mask = masks[frame_idx];
                int sum = mask_to_sum(mask) % 60;

                if (sum != s) { valid = 0; break; }

                int idx = find_combo_idx(s, mask);
                if (idx < 0) { valid = 0; break; }

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

            uint64_t k3_val = 0;
            for (int i = 0; i < n3; i++) {
                k3_val = k3_val * 3 + k3_digits[i];
            }
            uint64_t k_candidate = k3_val * 1073741824ULL + (k2_bits | k4_bits);

            int matches = verify_k(k_candidate, masks, rot, params);
            if (matches >= 54) {
                *out_k = k_candidate;
                return rot;
            }
        }
        return -1;
    }

    /* For custom periods, use brute-force search */
    for (int rot = 0; rot < 60; rot++) {
        /* First verify rotation is valid (sums match seconds) */
        int valid = 1;
        for (int s = 0; s < 60 && valid; s++) {
            int frame_idx = (s + rot) % 60;
            int sum = mask_to_sum(masks[frame_idx]) % 60;
            if (sum != s) valid = 0;
        }
        if (!valid) continue;

        /* Try all k values for this period */
        for (uint64_t k = 0; k < params->period; k++) {
            int matches = verify_k(k, masks, rot, params);
            if (matches >= 54) {
                *out_k = k;
                return rot;
            }
        }
    }

    return -1;
}

int inverse_extended(uint8_t (*observed_minutes)[60], int num_minutes,
                     const clock_params_t *params,
                     uint64_t *out_elapsed, uint64_t *out_variant) {
    if (num_minutes < 1) return -1;

    /* First, decode the first minute to get base k */
    uint64_t k;
    int rot = inverse_minute(observed_minutes[0], params, &k);
    if (rot < 0) return -1;

    /* If no variants, we're done */
    if (params->num_variants <= 1) {
        *out_elapsed = k * 60 + (60 - rot) % 60;
        *out_variant = 0;
        return 0;
    }

    /* TODO: Try each variant, find best match across all observed minutes */
    /* For now, return variant 0 */
    *out_elapsed = k * 60 + (60 - rot) % 60;
    *out_variant = 0;
    return 0;
}
