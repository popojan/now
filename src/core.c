/*
 * core.c - Mathematical core implementation for "now" clock
 */

#include "core.h"
#include <stdio.h>
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
    params->sig_period = 0;
    params->sig_value = 0;
}

uint64_t gcd(uint64_t a, uint64_t b) {
    while (b) {
        uint64_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

int is_coprime_60(uint64_t n) {
    return gcd(n, 60) == 1;
}

/* Original perm_index algorithm (compatible with web version) */
static int perm_index_original(uint64_t k, int s) {
    k = k % PERIOD_ORIGINAL_MINUTES;

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

int perm_index(uint64_t t, const clock_params_t *params) {
    int s = (int)(t % 60);
    uint64_t k = t / 60;
    int m = COMBO_CNT[s];

    if (m == 1) return 0;

    /* Encode signature N into k: combined = k * P + N */
    uint64_t k_combined = k;
    if (params->sig_period > 0) {
        k_combined = k * params->sig_period + params->sig_value;
    }

    return perm_index_original(k_combined, s);
}

uint8_t get_mask(uint64_t t, const clock_params_t *params) {
    int s = (int)(t % 60);
    int idx = perm_index(t, params);
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

/* ============ Signatures ============ */

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

uint64_t get_signature(uint64_t t, uint64_t period) {
    return t % period;
}

/* ============ Inverse (reconstruction) ============ */

/* Verify k_combined against observations spanning two minutes.
 * When first_sec > 0: seconds 0..first_sec-1 from minute k,
 *                     seconds first_sec..59 from minute k-1.
 * For signature mode: k_combined difference between minutes is P (not 1).
 * Returns match count (0-60). */
static int verify_k_spanning(uint64_t k_combined, uint8_t masks[60], int rot,
                             int first_sec, const clock_params_t *params) {
    /* Step between consecutive minutes in k_combined space */
    uint64_t step = (params->sig_period > 0) ? params->sig_period : 1;
    uint64_t k_combined_prev = (k_combined >= step) ? k_combined - step
                               : PERIOD_ORIGINAL_MINUTES - step + k_combined;
    int matches = 0;

    for (int s = 0; s < 60; s++) {
        int frame_idx = (s + rot) % 60;
        uint64_t k_eff;
        if (first_sec == 0) {
            k_eff = k_combined;
        } else if (s < first_sec) {
            k_eff = k_combined;       /* Seconds 0..first_sec-1 from minute k */
        } else {
            k_eff = k_combined_prev;  /* Seconds first_sec..59 from minute k-1 */
        }
        int expected_idx = perm_index_original(k_eff, s);
        uint8_t expected_mask = COMBOS[s][expected_idx];
        if (expected_mask == masks[frame_idx]) matches++;
    }
    return matches;
}

/* Exact reconstruction of rest (k2|k4) using bit-by-bit subtraction.
 * Given: K_known bits, L_known bits, P, relationship L_rest = K_rest - P
 * Returns K_rest exactly. Sets borrow_out for k3 calculation. */
static uint32_t reconstruct_rest_exact(int first_sec, uint32_t K_known, uint32_t L_known,
                                       uint32_t P_rest, int *borrow_out) {
    uint32_t K_rest = 0;

    /* Compute bit masks: which positions are from K vs L */
    uint32_t K_mask = 0, L_mask = 0;
    int n2 = 0, n4 = 0;

    for (int s = 0; s < 60; s++) {
        int m = COMBO_CNT[s];
        int is_K = (first_sec == 0) || (s < first_sec);

        if (m == 2) {
            int bit_pos = 29 - n2;
            if (is_K) K_mask |= (1U << bit_pos);
            else L_mask |= (1U << bit_pos);
            n2++;
        } else if (m == 4) {
            int bit_pos = 10 - 2 * n4;
            if (is_K) K_mask |= (3U << bit_pos);
            else L_mask |= (3U << bit_pos);
            n4++;
        }
    }

    /* Solve bit by bit from LSB to MSB using subtraction relationship:
     * L_rest = K_rest - P_rest (mod 2^30)
     * If we know K_bit: use it directly
     * If we know L_bit: K_bit = L_bit XOR P_bit XOR borrow */
    int borrow = 0;
    for (int i = 0; i < 30; i++) {
        int P_bit = (P_rest >> i) & 1;
        int K_bit;

        if ((K_mask >> i) & 1) {
            /* We know K_bit directly */
            K_bit = (K_known >> i) & 1;
        } else {
            /* We know L_bit, compute K_bit: K = L + P + borrow (in subtraction sense) */
            int L_bit = (L_known >> i) & 1;
            K_bit = (L_bit + P_bit + borrow) & 1;
        }

        K_rest |= ((uint32_t)K_bit << i);

        /* Compute next borrow: borrow if K_bit < P_bit + current_borrow */
        int sum = P_bit + borrow;
        borrow = (K_bit < sum) ? 1 : 0;
    }

    *borrow_out = borrow;
    return K_rest;
}

int inverse_time(uint8_t masks[60], const clock_params_t *params,
                 uint64_t *out_t, uint64_t *out_sig) {

    uint64_t P = (params->sig_period > 0) ? params->sig_period : 1;
    uint32_t P_rest = (uint32_t)(P % (1ULL << 30));
    uint64_t P_k3 = P / (1ULL << 30);  /* How P affects k3 part */

    /* Try all 60 rotations to find the starting second */
    for (int rot = 0; rot < 60; rot++) {
        /* First verify rotation is valid (sums match seconds) */
        int valid = 1;
        for (int s = 0; s < 60 && valid; s++) {
            int frame_idx = (s + rot) % 60;
            int sum = mask_to_sum(masks[frame_idx]) % 60;
            if (sum != s) valid = 0;
        }
        if (!valid) continue;

        int first_sec = (rot == 0) ? 0 : (60 - rot);

        /* Extract bits separately for each minute:
         * - K bits from seconds 0..first_sec-1 (minute k)
         * - L bits from seconds first_sec..59 (minute k-1, value K-P)
         */
        uint32_t K_rest_known = 0, L_rest_known = 0;
        int K_k3[16] = {0}, L_k3[16] = {0};
        int K_k3_mask[16] = {0};  /* 1 if we know this k3 digit from K */
        int n2 = 0, n3 = 0, n4 = 0;
        valid = 1;

        for (int s = 0; s < 60 && valid; s++) {
            int frame_idx = (s + rot) % 60;
            uint8_t mask = masks[frame_idx];

            int idx = find_combo_idx(s, mask);
            if (idx < 0) { valid = 0; break; }

            int m = COMBO_CNT[s];
            if (m == 1) continue;

            int is_from_K = (first_sec == 0) || (s < first_sec);

            if (m == 2) {
                if (is_from_K)
                    K_rest_known |= ((uint32_t)idx << (29 - n2));
                else
                    L_rest_known |= ((uint32_t)idx << (29 - n2));
                n2++;
            } else if (m == 3) {
                if (n3 < 16) {
                    if (is_from_K) {
                        K_k3[n3] = idx;
                        K_k3_mask[n3] = 1;
                    } else {
                        L_k3[n3] = idx;
                        K_k3_mask[n3] = 0;
                    }
                }
                n3++;
            } else if (m == 4) {
                if (is_from_K)
                    K_rest_known |= ((uint32_t)idx << (10 - 2*n4));
                else
                    L_rest_known |= ((uint32_t)idx << (10 - 2*n4));
                n4++;
            }
        }

        if (!valid) continue;

        /* No crossing: all data from single minute - exact reconstruction */
        if (first_sec == 0) {
            uint64_t K_k3_val = 0;
            for (int i = 0; i < 16; i++) {
                K_k3_val = K_k3_val * 3 + K_k3[i];
            }
            uint64_t k_combined = K_k3_val * (1ULL << 30) + K_rest_known;

            if (verify_k_spanning(k_combined, masks, rot, first_sec, params) == 60) {
                uint64_t k_original = k_combined;
                uint64_t sig_decoded = 0;
                if (params->sig_period > 0) {
                    sig_decoded = k_combined % params->sig_period;
                    k_original = k_combined / params->sig_period;
                }
                *out_t = k_original * 60;
                if (out_sig) *out_sig = sig_decoded;
                return rot;
            }
            continue;
        }

        /* Crossing case: exact reconstruction using relationship L = K - P */

        /* Step 1: Exact reconstruction of rest part */
        int borrow_from_rest;
        uint32_t K_rest = reconstruct_rest_exact(first_sec, K_rest_known,
                                                  L_rest_known, P_rest,
                                                  &borrow_from_rest);

        /* Step 2: For k3 part, enumerate unknown digits
         * L_k3_val = K_k3_val - borrow_from_rest - P_k3
         * We know some digits of K_k3, some of L_k3.
         * Try all combinations of unknown K_k3 digits. */

        int num_unknown = 0;
        int unknown_pos[16];
        for (int i = 0; i < 16; i++) {
            if (!K_k3_mask[i]) {
                unknown_pos[num_unknown++] = i;
            }
        }

        /* Enumerate all 3^num_unknown combinations */
        int num_combos = 1;
        for (int i = 0; i < num_unknown; i++) num_combos *= 3;

        int found = 0;
        uint64_t found_k = 0;

        for (int combo = 0; combo < num_combos && !found; combo++) {
            /* Set unknown K_k3 digits based on combo */
            int temp = combo;
            for (int i = 0; i < num_unknown; i++) {
                K_k3[unknown_pos[i]] = temp % 3;
                temp /= 3;
            }

            /* Compute K_k3_val */
            uint64_t K_k3_val = 0;
            for (int i = 0; i < 16; i++) {
                K_k3_val = K_k3_val * 3 + K_k3[i];
            }

            /* Compute L_k3_val = K_k3_val - borrow - P_k3 */
            uint64_t total_borrow = borrow_from_rest + P_k3;
            if (K_k3_val < total_borrow) continue;  /* Would underflow */
            uint64_t L_k3_val = K_k3_val - total_borrow;

            /* Extract L_k3 digits and verify against known L_k3 */
            int L_k3_digits[16];
            uint64_t temp_val = L_k3_val;
            for (int i = 15; i >= 0; i--) {
                L_k3_digits[i] = temp_val % 3;
                temp_val /= 3;
            }

            int match = 1;
            for (int i = 0; i < 16 && match; i++) {
                if (!K_k3_mask[i]) {
                    /* This digit comes from L, verify it */
                    if (L_k3_digits[i] != L_k3[i]) match = 0;
                }
            }

            if (match) {
                uint64_t k_combined = K_k3_val * (1ULL << 30) + K_rest;
                if (verify_k_spanning(k_combined, masks, rot, first_sec, params) == 60) {
                    found = 1;
                    found_k = k_combined;
                }
            }
        }

        if (found) {
            uint64_t k_original = found_k;
            uint64_t sig_decoded = 0;
            if (params->sig_period > 0) {
                sig_decoded = found_k % params->sig_period;
                k_original = found_k / params->sig_period;
            }
            uint64_t first_min = (k_original > 0) ? k_original - 1 : k_original;
            *out_t = first_min * 60 + first_sec;
            if (out_sig) *out_sig = sig_decoded;
            return rot;
        }
    }

    return -1;
}

uint64_t detect_signature_period(uint8_t *masks, int num_frames) {
    if (num_frames < 120) return 0;  /* Need at least 2 minutes */

    /* Look for repeating patterns in variant selections */
    /* For each candidate period P (coprime with 60), check if pattern repeats */
    uint64_t candidates[] = {7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 49, 53, 59, 61, 67, 71, 73, 77, 79, 83, 89, 97};
    int num_candidates = sizeof(candidates) / sizeof(candidates[0]);

    for (int c = 0; c < num_candidates; c++) {
        uint64_t P = candidates[c];
        if ((int)(2 * P) > num_frames) continue;

        /* Check if masks[i] pattern repeats with period P */
        int matches = 0;
        int checks = 0;
        for (int i = 0; i + (int)P < num_frames; i++) {
            /* Only check seconds with multiple variants */
            int s = i % 60;
            if (COMBO_CNT[s] > 1) {
                if (masks[i] == masks[i + (int)P]) matches++;
                checks++;
            }
        }

        /* If >90% match, likely found the period */
        if (checks > 0 && matches * 10 >= checks * 9) {
            return P;
        }
    }

    return 0;
}
