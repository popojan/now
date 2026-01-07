/*
 * core8.h - Mathematical core for 8-bit "now" Mondrian clock
 *
 * 8 cells: 1, 2, 4, 6, 12, 15, 20, 30 (sum = 90)
 * Period: 3^8 × 4^28 × 5^24 ≈ 2.8×10^37 minutes ≈ 5×10^31 years
 *
 * Distribution: 8 seconds with 3 combos, 28 with 4, 24 with 5
 * Entropy: 7.98 bits/second (99.8% efficiency)
 */

#ifndef NOW_CORE8_H
#define NOW_CORE8_H

#include <stdint.h>

/* 128-bit integer type for period arithmetic */
#ifdef __SIZEOF_INT128__
typedef unsigned __int128 uint128_t;
#define HAS_INT128 1
#else
/* Fallback: use struct for 128-bit */
typedef struct {
    uint64_t lo;
    uint64_t hi;
} uint128_t;
#define HAS_INT128 0
#endif

/* Period: 3^8 × 4^28 × 5^24 minutes
 * = 6561 × 72057594037927936 × 59604644775390625
 * = 28179280429056000000000000000000000000 (38 digits)
 * ≈ 2.818×10^37 minutes ≈ 5.36×10^31 years */
#define PERIOD8_M3 8    /* Exponent for base 3 */
#define PERIOD8_M4 28   /* Exponent for base 4 */
#define PERIOD8_M5 24   /* Exponent for base 5 */
#define PERIOD8_STR "28179280429056000000000000000000000000"

/* Which seconds have which multiplicity */
extern const int SEC_M3[8];   /* Seconds with 3 combinations */
extern const int SEC_M4[28];  /* Seconds with 4 combinations */
extern const int SEC_M5[24];  /* Seconds with 5 combinations */

/* Cell combinations for each second - stored as bitmask
 * Bits: 7=30, 6=20, 5=15, 4=12, 3=6, 2=4, 1=2, 0=1 */
extern const uint8_t COMBOS8[60][5];
extern const uint8_t COMBO8_CNT[60];

/* 128-bit arithmetic helpers */
#if HAS_INT128
#define U128_ZERO ((uint128_t)0)
#define U128_FROM_U64(x) ((uint128_t)(x))
#define U128_LO(x) ((uint64_t)(x))
#define U128_HI(x) ((uint64_t)((x) >> 64))
#define U128_ADD(a, b) ((a) + (b))
#define U128_SUB(a, b) ((a) - (b))
#define U128_MUL(a, b) ((a) * (b))
#define U128_DIV(a, b) ((a) / (b))
#define U128_MOD(a, b) ((a) % (b))
#define U128_LT(a, b) ((a) < (b))
#define U128_EQ(a, b) ((a) == (b))
#else
/* Software 128-bit arithmetic */
uint128_t u128_from_u64(uint64_t x);
uint128_t u128_add(uint128_t a, uint128_t b);
uint128_t u128_sub(uint128_t a, uint128_t b);
uint128_t u128_mul_u64(uint128_t a, uint64_t b);
uint128_t u128_div_u64(uint128_t a, uint64_t b, uint64_t *rem);
uint64_t u128_mod_u64(uint128_t a, uint64_t b);
int u128_lt(uint128_t a, uint128_t b);
int u128_eq(uint128_t a, uint128_t b);
#define U128_ZERO u128_from_u64(0)
#define U128_FROM_U64(x) u128_from_u64(x)
#define U128_LO(x) ((x).lo)
#define U128_HI(x) ((x).hi)
#define U128_ADD(a, b) u128_add(a, b)
#define U128_SUB(a, b) u128_sub(a, b)
#define U128_MUL(a, b) u128_mul_u64(a, b)
#define U128_DIV(a, b) u128_div_u64(a, b, NULL)
#define U128_MOD(a, b) u128_mod_u64(a, b)
#define U128_LT(a, b) u128_lt(a, b)
#define U128_EQ(a, b) u128_eq(a, b)
#endif

/* Forward declaration - use clock_params_t from core.h */
#include "core.h"

/* 128-bit string conversion */
uint128_t str_to_u128(const char *s);
void u128_to_str(uint128_t n, char *buf, int bufsize);

/* 128-bit division with quotient and remainder */
void u128_divmod(uint128_t a, uint128_t b, uint128_t *quot, uint128_t *rem);

/* 8-bit mode parameters (supports full 128-bit P and N) */
typedef struct {
    uint128_t sig_period;   /* P: signature period (0 = disabled) */
    uint128_t sig_value;    /* N: signature value (0 to P-1) */
} clock8_params_t;

void clock8_params_init(clock8_params_t *params);

/* Compute permutation index for elapsed time t (in seconds)
 * Returns which combo from COMBOS8[s] to use, where s = t % 60
 * Uses params for signature encoding (P, N) */
int perm8_index(uint128_t t, const clock8_params_t *params);

/* Get mask for a specific time t (in seconds) */
uint8_t get_mask8(uint128_t t, const clock8_params_t *params);

/* Convert 8-bit mask to sum (the second value it represents, mod 60) */
int mask8_to_sum(uint8_t mask);

/* Find which combo index produces the given mask for second s
 * Returns -1 if not found */
int find_combo8_idx(int s, uint8_t mask);

/* Inverse: reconstruct time from 60 masks
 * Returns rotation (which mask is second 0), or -1 on failure
 * out_t: reconstructed elapsed time in seconds
 * out_sig: signature value (k_combined % P) if P > 0 */
int inverse8_time(uint8_t masks[60], const clock8_params_t *params,
                  uint128_t *out_t, uint128_t *out_sig);

#endif /* NOW_CORE8_H */
