/*
 * core.h - Mathematical core for "now" Mondrian clock
 *
 * Pure algorithms, no I/O. Handles:
 * - Permutation generation (which combo to show for each second)
 * - Divisor/signature calculation
 * - Inverse mapping (masks -> elapsed time)
 * - Kolmogorov variants (algorithm variants that extend period)
 */

#ifndef NOW_CORE_H
#define NOW_CORE_H

#include <stdint.h>

/* Original period: 2^30 * 3^16 minutes */
#define PERIOD_ORIGINAL 46221064723759104ULL

/* Cell combinations for each second (0-59)
 * Stored as bitmask: bits for cells 20,15,12,6,4,2,1 */
extern const uint8_t COMBOS[60][4];
extern const uint8_t COMBO_CNT[60];

/* Clock parameters */
typedef struct {
    uint64_t period;        /* Base period in minutes (default: PERIOD_ORIGINAL) */
    uint64_t num_variants;  /* Kolmogorov variants (1 = disabled, >1 = extends period) */
    uint64_t variant;       /* Current variant (0 to num_variants-1) */
} clock_params_t;

/* Initialize with defaults (original now behavior) */
void clock_params_init(clock_params_t *params);

/* Compute permutation index for second s at minute k
 * Returns which combo from COMBOS[s] to use */
int perm_index(uint64_t k, int s, const clock_params_t *params);

/* Get mask for a specific second at minute k */
uint8_t get_mask(uint64_t k, int s, const clock_params_t *params);

/* Convert mask to sum (the second value it represents) */
int mask_to_sum(uint8_t mask);

/* Find which combo index produces the given mask for second s
 * Returns -1 if not found */
int find_combo_idx(int s, uint8_t mask);

/* ============ Divisors and Signatures ============ */

/* Get all divisors of n, sorted ascending
 * Returns count of divisors, stores in out[] (max max_out) */
int get_divisors(uint64_t n, uint64_t *out, int max_out);

/* Get prime factorization of n
 * Returns count, stores primes in primes[], exponents in exps[] */
int factorize(uint64_t n, uint64_t *primes, int *exps, int max_factors);

/* Get signature value for a specific divisor
 * sig[d] represents position within the d-minute cycle */
uint64_t get_signature(uint64_t elapsed_minutes, uint64_t divisor);

/* Get all signature values for all divisors of period
 * sigs[i] corresponds to divisors[i] */
void get_all_signatures(uint64_t elapsed_minutes, uint64_t period,
                        uint64_t *divisors, uint64_t *sigs, int num_divisors);

/* ============ Inverse (reconstruction) ============ */

/* Reconstruct minute k from 60 observed masks
 * Returns rotation (which mask is second 0), or -1 on failure
 * Stores reconstructed k in *out_k */
int inverse_minute(uint8_t masks[60], const clock_params_t *params, uint64_t *out_k);

/* Extended inverse: also determine variant from multiple minutes of observation
 * observed_minutes: array of 60-mask arrays for each minute
 * num_minutes: how many minutes observed
 * Returns 0 on success, -1 on failure */
int inverse_extended(uint8_t (*observed_minutes)[60], int num_minutes,
                     const clock_params_t *params,
                     uint64_t *out_elapsed, uint64_t *out_variant);

#endif /* NOW_CORE_H */
