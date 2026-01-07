/*
 * core8.c - Mathematical core implementation for 8-bit "now" clock
 *
 * 8 cells: 1, 2, 4, 6, 12, 15, 20, 30 (sum = 90)
 * Period: 3^8 × 4^28 × 5^24 minutes
 */

#include "core8.h"
#include <string.h>

/* Which seconds have which multiplicity */
const int SEC_M3[8] = {1, 14, 16, 29, 31, 44, 46, 59};
const int SEC_M4[28] = {0, 2, 4, 5, 10, 11, 13, 15, 17, 19, 20, 25, 26, 28, 30, 32, 34, 35, 40, 41, 43, 45, 47, 49, 50, 55, 56, 58};
const int SEC_M5[24] = {3, 6, 7, 8, 9, 12, 18, 21, 22, 23, 24, 27, 33, 36, 37, 38, 39, 42, 48, 51, 52, 53, 54, 57};

/* Cell combinations for each second - stored as bitmask
 * Bits: 7=30, 6=20, 5=15, 4=12, 3=6, 2=4, 1=2, 0=1 */
const uint8_t COMBOS8[60][5] = {
    {0x00,0x7F,0xB3,0xCC},  /* s=0, m=4 */
    {0x01,0xB4,0xCD},  /* s=1, m=3 */
    {0x02,0xB5,0xCE,0xD0},  /* s=2, m=4 */
    {0x03,0xB6,0xB8,0xCF,0xD1},  /* s=3, m=5 */
    {0x04,0xB7,0xB9,0xD2},  /* s=4, m=4 */
    {0x05,0xBA,0xD3,0xE0},  /* s=5, m=4 */
    {0x06,0x08,0xBB,0xD4,0xE1},  /* s=6, m=5 */
    {0x07,0x09,0xBC,0xD5,0xE2},  /* s=7, m=5 */
    {0x0A,0xBD,0xD6,0xD8,0xE3},  /* s=8, m=5 */
    {0x0B,0xBE,0xD7,0xD9,0xE4},  /* s=9, m=5 */
    {0x0C,0xBF,0xDA,0xE5},  /* s=10, m=4 */
    {0x0D,0xDB,0xE6,0xE8},  /* s=11, m=4 */
    {0x0E,0x10,0xDC,0xE7,0xE9},  /* s=12, m=5 */
    {0x0F,0x11,0xDD,0xEA},  /* s=13, m=4 */
    {0x12,0xDE,0xEB},  /* s=14, m=3 */
    {0x13,0x20,0xDF,0xEC},  /* s=15, m=4 */
    {0x14,0x21,0xED},  /* s=16, m=3 */
    {0x15,0x22,0xEE,0xF0},  /* s=17, m=4 */
    {0x16,0x18,0x23,0xEF,0xF1},  /* s=18, m=5 */
    {0x17,0x19,0x24,0xF2},  /* s=19, m=4 */
    {0x1A,0x25,0x40,0xF3},  /* s=20, m=4 */
    {0x1B,0x26,0x28,0x41,0xF4},  /* s=21, m=5 */
    {0x1C,0x27,0x29,0x42,0xF5},  /* s=22, m=5 */
    {0x1D,0x2A,0x43,0xF6,0xF8},  /* s=23, m=5 */
    {0x1E,0x2B,0x44,0xF7,0xF9},  /* s=24, m=5 */
    {0x1F,0x2C,0x45,0xFA},  /* s=25, m=4 */
    {0x2D,0x46,0x48,0xFB},  /* s=26, m=4 */
    {0x2E,0x30,0x47,0x49,0xFC},  /* s=27, m=5 */
    {0x2F,0x31,0x4A,0xFD},  /* s=28, m=4 */
    {0x32,0x4B,0xFE},  /* s=29, m=3 */
    {0x33,0x4C,0x80,0xFF},  /* s=30, m=4 */
    {0x34,0x4D,0x81},  /* s=31, m=3 */
    {0x35,0x4E,0x50,0x82},  /* s=32, m=4 */
    {0x36,0x38,0x4F,0x51,0x83},  /* s=33, m=5 */
    {0x37,0x39,0x52,0x84},  /* s=34, m=4 */
    {0x3A,0x53,0x60,0x85},  /* s=35, m=4 */
    {0x3B,0x54,0x61,0x86,0x88},  /* s=36, m=5 */
    {0x3C,0x55,0x62,0x87,0x89},  /* s=37, m=5 */
    {0x3D,0x56,0x58,0x63,0x8A},  /* s=38, m=5 */
    {0x3E,0x57,0x59,0x64,0x8B},  /* s=39, m=5 */
    {0x3F,0x5A,0x65,0x8C},  /* s=40, m=4 */
    {0x5B,0x66,0x68,0x8D},  /* s=41, m=4 */
    {0x5C,0x67,0x69,0x8E,0x90},  /* s=42, m=5 */
    {0x5D,0x6A,0x8F,0x91},  /* s=43, m=4 */
    {0x5E,0x6B,0x92},  /* s=44, m=3 */
    {0x5F,0x6C,0x93,0xA0},  /* s=45, m=4 */
    {0x6D,0x94,0xA1},  /* s=46, m=3 */
    {0x6E,0x70,0x95,0xA2},  /* s=47, m=4 */
    {0x6F,0x71,0x96,0x98,0xA3},  /* s=48, m=5 */
    {0x72,0x97,0x99,0xA4},  /* s=49, m=4 */
    {0x73,0x9A,0xA5,0xC0},  /* s=50, m=4 */
    {0x74,0x9B,0xA6,0xA8,0xC1},  /* s=51, m=5 */
    {0x75,0x9C,0xA7,0xA9,0xC2},  /* s=52, m=5 */
    {0x76,0x78,0x9D,0xAA,0xC3},  /* s=53, m=5 */
    {0x77,0x79,0x9E,0xAB,0xC4},  /* s=54, m=5 */
    {0x7A,0x9F,0xAC,0xC5},  /* s=55, m=4 */
    {0x7B,0xAD,0xC6,0xC8},  /* s=56, m=4 */
    {0x7C,0xAE,0xB0,0xC7,0xC9},  /* s=57, m=5 */
    {0x7D,0xAF,0xB1,0xCA},  /* s=58, m=4 */
    {0x7E,0xB2,0xCB},  /* s=59, m=3 */
};

const uint8_t COMBO8_CNT[60] = {
    4,3,4,5,4,4,5,5,5,5,4,4,5,4,3,4,3,4,5,4,4,5,5,5,5,4,4,5,4,3,4,3,4,5,4,4,5,5,5,5,4,4,5,4,3,4,3,4,5,4,4,5,5,5,5,4,4,5,4,3
};

void clock8_params_init(clock8_params_t *params) {
    params->sig_period = U128_ZERO;
    params->sig_value = U128_ZERO;
}

/* Parse decimal string to uint128_t */
uint128_t str_to_u128(const char *s) {
    uint128_t result = U128_ZERO;
    while (*s >= '0' && *s <= '9') {
        result = U128_MUL(result, 10);
        result = U128_ADD(result, U128_FROM_U64(*s - '0'));
        s++;
    }
    return result;
}

/* Convert uint128_t to decimal string */
void u128_to_str(uint128_t n, char *buf, int bufsize) {
    if (bufsize < 2) { buf[0] = '\0'; return; }

    /* Handle zero */
    if (U128_EQ(n, U128_ZERO)) {
        buf[0] = '0';
        buf[1] = '\0';
        return;
    }

    /* Build string in reverse */
    char temp[50];
    int pos = 0;

    while (!U128_EQ(n, U128_ZERO) && pos < 49) {
        uint64_t digit = U128_MOD(n, 10);
        temp[pos++] = '0' + digit;
        n = U128_DIV(n, 10);
    }

    /* Reverse into output buffer */
    int i;
    for (i = 0; i < pos && i < bufsize - 1; i++) {
        buf[i] = temp[pos - 1 - i];
    }
    buf[i] = '\0';
}

/* Software 128-bit arithmetic (fallback if no __int128) */
#if !HAS_INT128

uint128_t u128_from_u64(uint64_t x) {
    uint128_t r = {x, 0};
    return r;
}

uint128_t u128_add(uint128_t a, uint128_t b) {
    uint128_t r;
    r.lo = a.lo + b.lo;
    r.hi = a.hi + b.hi + (r.lo < a.lo ? 1 : 0);
    return r;
}

uint128_t u128_sub(uint128_t a, uint128_t b) {
    uint128_t r;
    r.lo = a.lo - b.lo;
    r.hi = a.hi - b.hi - (a.lo < b.lo ? 1 : 0);
    return r;
}

uint128_t u128_mul_u64(uint128_t a, uint64_t b) {
    /* Split into 32-bit parts for overflow-safe multiplication */
    uint64_t a_lo = a.lo & 0xFFFFFFFFULL;
    uint64_t a_hi = a.lo >> 32;
    uint64_t b_lo = b & 0xFFFFFFFFULL;
    uint64_t b_hi = b >> 32;

    uint64_t p0 = a_lo * b_lo;
    uint64_t p1 = a_lo * b_hi;
    uint64_t p2 = a_hi * b_lo;
    uint64_t p3 = a_hi * b_hi;

    uint64_t carry = (p0 >> 32) + (p1 & 0xFFFFFFFFULL) + (p2 & 0xFFFFFFFFULL);

    uint128_t r;
    r.lo = (p0 & 0xFFFFFFFFULL) | (carry << 32);
    r.hi = a.hi * b + p3 + (p1 >> 32) + (p2 >> 32) + (carry >> 32);
    return r;
}

uint128_t u128_div_u64(uint128_t a, uint64_t b, uint64_t *rem) {
    /* Simple long division */
    uint128_t q = {0, 0};
    uint64_t r = 0;

    for (int i = 127; i >= 0; i--) {
        r <<= 1;
        if (i >= 64) {
            r |= (a.hi >> (i - 64)) & 1;
        } else {
            r |= (a.lo >> i) & 1;
        }
        if (r >= b) {
            r -= b;
            if (i >= 64) {
                q.hi |= 1ULL << (i - 64);
            } else {
                q.lo |= 1ULL << i;
            }
        }
    }

    if (rem) *rem = r;
    return q;
}

uint64_t u128_mod_u64(uint128_t a, uint64_t b) {
    uint64_t rem;
    u128_div_u64(a, b, &rem);
    return rem;
}

int u128_lt(uint128_t a, uint128_t b) {
    if (a.hi != b.hi) return a.hi < b.hi;
    return a.lo < b.lo;
}

int u128_eq(uint128_t a, uint128_t b) {
    return a.hi == b.hi && a.lo == b.lo;
}

/* Full 128x128 -> 128 multiplication (ignores overflow beyond 128 bits) */
uint128_t u128_mul_u128(uint128_t a, uint128_t b) {
    /* a * b = (a_hi * 2^64 + a_lo) * (b_hi * 2^64 + b_lo)
     *       = a_hi*b_hi*2^128 + (a_hi*b_lo + a_lo*b_hi)*2^64 + a_lo*b_lo
     * We only keep the lower 128 bits, so ignore a_hi*b_hi term */
    uint128_t result = u128_mul_u64(u128_from_u64(a.lo), b.lo);
    result.hi += a.lo * b.hi + a.hi * b.lo;
    return result;
}

#endif /* !HAS_INT128 */

/*
 * perm8_index - compute which combination to use for time t
 *
 * Encoding scheme:
 *   k_raw = t / 60 (logical minute)
 *   k_combined = k_raw * P + N (with signature encoding)
 *   k_combined is encoded as: k = k5 * (3^8 * 4^28) + k4 * (3^8) + k3
 *   where:
 *     k3: base-3 number with 8 digits (for m=3 seconds)
 *     k4: base-4 number with 28 digits (for m=4 seconds)
 *     k5: base-5 number with 24 digits (for m=5 seconds)
 */
int perm8_index(uint128_t t, const clock8_params_t *params) {
    int s = (int)U128_MOD(t, 60);
    uint128_t k_raw = U128_DIV(t, 60);  /* Logical minute */

    /* Apply signature encoding: k_combined = k_raw * P + N */
    uint128_t k;
    uint128_t one = U128_FROM_U64(1);
    if (params && U128_LT(one, params->sig_period)) {
        /* k = k_raw * P + N (full 128-bit arithmetic) */
#if HAS_INT128
        k = k_raw * params->sig_period + params->sig_value;
#else
        k = u128_mul_u128(k_raw, params->sig_period);
        k = U128_ADD(k, params->sig_value);
#endif
    } else {
        k = k_raw;
    }

    int m = COMBO8_CNT[s];
    if (m < 3 || m > 5) return 0;  /* Should not happen */

    /* Count how many m=3/4/5 seconds come before s */
    int n3 = 0, n4 = 0, n5 = 0;
    for (int i = 0; i < s; i++) {
        int mi = COMBO8_CNT[i];
        if (mi == 3) n3++;
        else if (mi == 4) n4++;
        else if (mi == 5) n5++;
    }

    /* Extract the appropriate digit from k
     * k is structured as: k5_digits || k4_digits || k3_digits
     * where k3 is 8 base-3 digits, k4 is 28 base-4 digits, k5 is 24 base-5 digits
     *
     * k3_part = k % 3^8
     * k4_part = (k / 3^8) % 4^28
     * k5_part = (k / 3^8 / 4^28) % 5^24
     */

    /* 3^8 = 6561 */
    const uint64_t POW3_8 = 6561ULL;
    /* 4^28 = 2^56 = 72057594037927936 */
    const uint64_t POW4_28 = 72057594037927936ULL;

    if (m == 3) {
        /* Extract digit n3 from k3_part (base 3, 8 digits) */
        uint64_t k3_part = U128_MOD(k, POW3_8);
        /* Digit n3 (0-indexed from right): divide by 3^n3, then mod 3 */
        uint64_t divisor = 1;
        for (int i = 0; i < n3; i++) divisor *= 3;
        return (int)((k3_part / divisor) % 3);
    }

    if (m == 4) {
        /* Extract digit n4 from k4_part (base 4, 28 digits) */
        uint128_t k_shifted = U128_DIV(k, POW3_8);
        uint64_t k4_part = U128_MOD(k_shifted, POW4_28);
        /* Digit n4 (0-indexed from right): bit position n4*2 */
        int bit_pos = n4 * 2;
        return (int)((k4_part >> bit_pos) & 3);
    }

    if (m == 5) {
        /* Extract digit n5 from k5_part (base 5, 24 digits) */
        /* 5^24 = 59604644775390625, fits in uint64 */
        const uint64_t POW5_24 = 59604644775390625ULL;

        uint128_t k_shifted = U128_DIV(k, POW3_8);
        k_shifted = U128_DIV(k_shifted, POW4_28);
        uint64_t k5_part = U128_MOD(k_shifted, POW5_24);

        /* Digit n5 (0-indexed from right): divide by 5^n5, then mod 5 */
        uint64_t divisor = 1;
        for (int i = 0; i < n5; i++) divisor *= 5;
        return (int)((k5_part / divisor) % 5);
    }

    return 0;
}

uint8_t get_mask8(uint128_t t, const clock8_params_t *params) {
    int s = (int)U128_MOD(t, 60);
    int idx = perm8_index(t, params);
    return COMBOS8[s][idx];
}

int mask8_to_sum(uint8_t mask) {
    int sum = 0;
    if (mask & 0x01) sum += 1;
    if (mask & 0x02) sum += 2;
    if (mask & 0x04) sum += 4;
    if (mask & 0x08) sum += 6;
    if (mask & 0x10) sum += 12;
    if (mask & 0x20) sum += 15;
    if (mask & 0x40) sum += 20;
    if (mask & 0x80) sum += 30;
    return sum % 60;
}

int find_combo8_idx(int s, uint8_t mask) {
    int cnt = COMBO8_CNT[s];
    for (int i = 0; i < cnt; i++) {
        if (COMBOS8[s][i] == mask) return i;
    }
    return -1;
}

/* 128-bit division helper for inverse */
#if HAS_INT128
void u128_divmod(uint128_t a, uint128_t b, uint128_t *quot, uint128_t *rem) {
    if (quot) *quot = a / b;
    if (rem) *rem = a % b;
}
#else
void u128_divmod(uint128_t a, uint128_t b, uint128_t *quot, uint128_t *rem) {
    /* Simple binary long division for 128-bit / 128-bit */
    uint128_t q = {0, 0};
    uint128_t r = {0, 0};

    for (int i = 127; i >= 0; i--) {
        /* r = r << 1 */
        r.hi = (r.hi << 1) | (r.lo >> 63);
        r.lo <<= 1;

        /* r |= bit i of a */
        if (i >= 64) {
            r.lo |= (a.hi >> (i - 64)) & 1;
        } else {
            r.lo |= (a.lo >> i) & 1;
        }

        /* if r >= b */
        if (!u128_lt(r, b)) {
            r = u128_sub(r, b);
            if (i >= 64) {
                q.hi |= 1ULL << (i - 64);
            } else {
                q.lo |= 1ULL << i;
            }
        }
    }
    if (quot) *quot = q;
    if (rem) *rem = r;
}
#endif

/*
 * inverse8_time - reconstruct time from 60 masks
 *
 * Full 128-bit reconstruction including k5 component.
 *
 * With signature encoding: k_combined = k_raw * P + N
 * So: N = k_combined % P, k_raw = k_combined / P
 * t = k_raw * 60 + rotation
 */
int inverse8_time(uint8_t masks[60], const clock8_params_t *params,
                  uint128_t *out_t, uint128_t *out_sig) {
    /* Constants */
    const uint64_t POW3_8 = 6561ULL;           /* 3^8 */
    const uint64_t POW4_28 = 72057594037927936ULL;  /* 4^28 = 2^56 */

    /* Try all 60 rotations */
    for (int rot = 0; rot < 60; rot++) {
        /* Verify sums match */
        int valid = 1;
        for (int s = 0; s < 60 && valid; s++) {
            int frame_idx = (s + rot) % 60;
            int sum = mask8_to_sum(masks[frame_idx]);
            if (sum != s) valid = 0;
        }
        if (!valid) continue;

        /* Extract digits for k3, k4, k5 */
        int k3_digits[8] = {0};
        int k4_digits[28] = {0};
        int k5_digits[24] = {0};
        int n3 = 0, n4 = 0, n5 = 0;

        valid = 1;
        for (int s = 0; s < 60 && valid; s++) {
            int frame_idx = (s + rot) % 60;
            uint8_t mask = masks[frame_idx];
            int idx = find_combo8_idx(s, mask);
            if (idx < 0) { valid = 0; break; }

            int m = COMBO8_CNT[s];
            if (m == 3) {
                k3_digits[n3++] = idx;
            } else if (m == 4) {
                k4_digits[n4++] = idx;
            } else if (m == 5) {
                k5_digits[n5++] = idx;
            }
        }

        if (!valid || n3 != 8 || n4 != 28 || n5 != 24) continue;

        /* Reconstruct k3, k4, k5 values (digits are LSB first) */
        uint64_t k3_val = 0;
        for (int i = 7; i >= 0; i--) {
            k3_val = k3_val * 3 + k3_digits[i];
        }

        uint64_t k4_val = 0;
        for (int i = 27; i >= 0; i--) {
            k4_val = k4_val * 4 + k4_digits[i];
        }

        uint64_t k5_val = 0;
        for (int i = 23; i >= 0; i--) {
            k5_val = k5_val * 5 + k5_digits[i];
        }

        /* k_combined = k5 * (4^28 * 3^8) + k4 * 3^8 + k3
         * Full 128-bit reconstruction */
        uint128_t k_combined = U128_FROM_U64(k3_val);

        /* Add k4 * 3^8 */
        uint128_t k4_contrib = U128_MUL(U128_FROM_U64(k4_val), POW3_8);
        k_combined = U128_ADD(k_combined, k4_contrib);

        /* Add k5 * 4^28 * 3^8 */
        if (k5_val > 0) {
            uint128_t k5_128 = U128_FROM_U64(k5_val);
            uint128_t k5_contrib = U128_MUL(k5_128, POW4_28);
            k5_contrib = U128_MUL(k5_contrib, POW3_8);
            k_combined = U128_ADD(k_combined, k5_contrib);
        }

        /* Extract signature and compute elapsed time */
        uint128_t k_raw, sig;
        uint128_t one = U128_FROM_U64(1);

        if (params && U128_LT(one, params->sig_period)) {
            /* k_raw = k_combined / P, sig = k_combined % P */
            u128_divmod(k_combined, params->sig_period, &k_raw, &sig);
        } else {
            sig = U128_ZERO;
            k_raw = k_combined;
        }

        /* t = k_raw * 60 + rot */
        uint128_t t = U128_MUL(k_raw, 60);
        t = U128_ADD(t, U128_FROM_U64(rot));

        *out_t = t;
        if (out_sig) *out_sig = sig;

        return rot;
    }

    return -1;
}
