/*
 * core.h - Mathematical core for "now" Mondrian clock
 *
 * Pure algorithms, no I/O. Handles:
 * - Permutation generation (which combo to show for each second)
 * - Signature encoding/decoding (custom values encoded in time structure)
 * - Inverse mapping (masks -> elapsed time + signatures)
 *
 * Time model:
 *   t = elapsed seconds since origin
 *   s = t mod 60 (visual second, shown by cell sum)
 *   k = t / 60 (logical minute, determines variant selection)
 *
 * Signature encoding:
 *   For period P coprime with 60: sig[P] = encoded value (0 to P-1)
 *   The signature modifies variant selection, independently of origin.
 *   Constraint: P must be coprime with 60 (no factors 2, 3, or 5)
 *   for signatures to be independent of visual second.
 */

#ifndef NOW_CORE_H
#define NOW_CORE_H

#include <stdint.h>

/* Original period: 2^30 * 3^16 minutes = ~88 billion years */
#define PERIOD_ORIGINAL_MINUTES 46221064723759104ULL
#define PERIOD_ORIGINAL_SECONDS (PERIOD_ORIGINAL_MINUTES * 60ULL)

/* Cell combinations for each second (0-59)
 * Stored as bitmask: bits for cells 20,15,12,6,4,2,1 */
extern const uint8_t COMBOS[60][4];
extern const uint8_t COMBO_CNT[60];

/* Clock parameters */
typedef struct {
    /* Signature encoding: combined = k * P + N
     * This encodes value N (0 to P-1) into each cycle of P minutes.
     * Period is reduced by factor P, but still huge for reasonable P.
     * Set sig_period = 0 to disable (original 88B-year behavior). */
    uint64_t sig_period;      /* Period P for signature (0 = disabled) */
    uint64_t sig_value;       /* Encoded value N (0 to sig_period-1) */
} clock_params_t;

/* Initialize with defaults (original behavior, no signature encoding) */
void clock_params_init(clock_params_t *params);

/* Check if n is coprime with 60 (valid for signature encoding) */
int is_coprime_60(uint64_t n);

/* GCD helper */
uint64_t gcd(uint64_t a, uint64_t b);

/* Compute permutation index for elapsed time t (in seconds)
 * Returns which combo from COMBOS[s] to use, where s = t % 60 */
int perm_index(uint64_t t, const clock_params_t *params);

/* Get mask for a specific time t (in seconds) */
uint8_t get_mask(uint64_t t, const clock_params_t *params);

/* Convert mask to sum (the second value it represents) */
int mask_to_sum(uint8_t mask);

/* Find which combo index produces the given mask for second s
 * Returns -1 if not found */
int find_combo_idx(int s, uint8_t mask);

/* ============ Signatures ============ */

/* Get prime factorization of n
 * Returns count, stores primes in primes[], exponents in exps[] */
int factorize(uint64_t n, uint64_t *primes, int *exps, int max_factors);

/* Get signature value: t mod period */
uint64_t get_signature(uint64_t t, uint64_t period);

/* ============ Inverse (reconstruction) ============ */

/* Reconstruct elapsed time t from 60 observed masks
 * Returns rotation (which mask is second 0), or -1 on failure
 * Stores reconstructed t in *out_t
 * If sig_period > 0, also extracts signature value */
int inverse_time(uint8_t masks[60], const clock_params_t *params,
                 uint64_t *out_t, uint64_t *out_sig);

/* Auto-detect signature period by observing pattern repetition
 * Requires observing at least 2*P seconds to detect period P
 * Returns detected period, or 0 if no repetition found */
uint64_t detect_signature_period(uint8_t *masks, int num_frames);

/* ============ Error Correction ============ */

/* All valid masks for a given second (sum) */
int get_valid_masks_for_sum(int sum, uint8_t *out_masks, int max_out);

/* Hamming distance between two masks */
int hamming_distance(uint8_t a, uint8_t b);

/* Find closest valid mask for target sum
 * Returns the valid mask with minimum Hamming distance to 'received'
 * Also returns the distance via *dist_out (can be NULL) */
uint8_t closest_valid_mask(uint8_t received, int target_sum, int *dist_out);

/* Error correction result for a frame */
typedef struct {
    uint8_t original;      /* Original received mask */
    uint8_t corrected;     /* Corrected mask (or original if no correction needed) */
    int expected_sum;      /* Expected sum from sequence */
    int received_sum;      /* Actually received sum */
    int distance;          /* Hamming distance of correction (0 = no error) */
    int is_anchor;         /* 1 if this frame was an anchor (trusted) */
} frame_correction_t;

/* Correct errors in a sequence of masks
 * Uses anchor-based algorithm:
 * 1. Find virtual_start = (sum - position) mod 60 for each frame
 * 2. Majority virtual_start defines the correct sequence
 * 3. Frames matching majority are anchors
 * 4. Other frames are corrected to closest valid mask
 *
 * Returns number of corrections made, -1 on failure
 * Results are stored in corrections[] array */
int correct_errors(uint8_t *masks, int num_frames,
                   frame_correction_t *corrections, int *anchor_count);

#endif /* NOW_CORE_H */
