// gcc -std=c23 -O2 -frounding-math -march=native -lm -o check_fa3 check_fa3.c -g -fopenmp -DIS_LOCAL=true -DN_Q=8192 -DN_KV=8192 -DQK_SCALE=1

#include "utils/bf16.h"
#include "utils/ex.h"
#include "utils/load_file.h"
#include "utils/minmax.h"
#include "utils/tc.h"

// TODO: rewrite MUFU.RCP properly.
#include "../ptx_math_recip.h"

#include <stdint.h>
#include <stdio.h>

#include <err.h>

#if !defined(IS_LOCAL) || !defined(N_Q) || !defined(N_KV) || !defined(QK_SCALE)
#error "Define IS_LOCAL, N_Q, N_KV, and QK_SCALE"
#endif

constexpr size_t N_Q_HEADS  = 32;
constexpr size_t N_KV_HEADS = 2;

constexpr size_t HEAD_DIM = 128;
constexpr size_t KV_TILE  = 128;
// The denominator accounts for the PackGQA optimization from the FA3 paper.
constexpr size_t Q_TILE   = 128 / (N_Q_HEADS/N_KV_HEADS);

// Not including the token itself.
constexpr size_t WINDOW_SIZE = 2047;

struct InOuts {
    bf16 q  [N_Q_HEADS] [N_Q]     [HEAD_DIM];
    bf16 k  [N_KV_HEADS][N_KV]    [HEAD_DIM];
    bf16 v  [N_KV_HEADS][HEAD_DIM][N_KV];
    bf16 out[N_Q_HEADS] [N_Q]     [HEAD_DIM];
};

typeof((struct InOuts){0}.out) OUR_OUT;

size_t sat_sub(size_t x, size_t y) {
    return x > y ? x - y : 0;
}

void compute_query_head(
    size_t qi,
    const bf16 q  [HEAD_DIM],
    bf16       out[HEAD_DIM],
    const bf16 ks [N_KV][HEAD_DIM],
    const bf16 vs [HEAD_DIM][N_KV],
    float qk_scale
) {
    constexpr size_t N_SUMS         = 4;
    constexpr size_t N_ELEM_PER_SUM = 2;

    float logits [KV_TILE];
    bf16  scores [KV_TILE];
    float out_raw[KV_TILE] = {};
    float l_max = -INFINITY;
    float s_sums[N_SUMS] = {};

    size_t last_kv_pos  = qi + N_KV - N_Q;
    size_t first_kv_pos = IS_LOCAL
                        ? sat_sub(last_kv_pos, WINDOW_SIZE)
                        : 0;
    size_t offset       = IS_LOCAL
                        ? sat_sub(first_kv_pos, qi%Q_TILE)
                        : 0;

    for (ptrdiff_t ti = (last_kv_pos - offset)/KV_TILE; ti >= 0; --ti) {
        size_t kv_tile_start = ti*KV_TILE + offset;

        // 1. Do the Q@K GEMM, update the score maximum.
        float prev_l_max = l_max;
        for (size_t ii = 0; ii < KV_TILE; ++ii) {
            size_t kvi = kv_tile_start + ii;
            float score = first_kv_pos <= kvi && kvi <= last_kv_pos
                        ? tc_bf16_fp32(0.0f, q, ks[kvi], HEAD_DIM)
                        : -INFINITY;
            logits[ii] = score;
            l_max = maxf(l_max, score);
        }

        // 2. Scale the previous score sums.
        float max_scaled = l_max * qk_scale;
        float score_scale = ex2((prev_l_max - l_max)*qk_scale);

        // 3. Exponentiate the scores, update the score sums.
        for (size_t ii = 0; ii < KV_TILE; ++ii) {
            float score = ex2(fmaf(logits[ii], qk_scale, -max_scaled));
            scores[ii] = to_bf16(score);

            size_t qq = ii / N_ELEM_PER_SUM;
            size_t rr = ii % N_ELEM_PER_SUM;
            size_t sum_idx = qq % N_SUMS;
            // First score is FMA-fused with sum rescaling.
            if (qq < N_SUMS && rr == 0) {
                s_sums[sum_idx] = fmaf(s_sums[sum_idx], score_scale, score);
            } else {
                s_sums[sum_idx] += score;
            }
        }

        // 4. Scale the output, do the Scores@V GEMM.
        for (size_t ii = 0; ii < HEAD_DIM; ++ii) {
            size_t n_gemm_elems = mins(N_KV - kv_tile_start, KV_TILE);
            out_raw[ii] = tc_bf16_fp32(
                out_raw[ii] * score_scale,
                scores,
                vs[ii] + kv_tile_start,
                n_gemm_elems
            );
        }
    }

    // Make the final output correction.
    float final_sum = (s_sums[0] + s_sums[2]) + (s_sums[1] + s_sums[3]);
    float final_sum_inv = ptxm_rcp_sm5x(final_sum);

    for (size_t ii = 0; ii < HEAD_DIM; ++ii) {
        out[ii] = to_bf16(out_raw[ii] * final_sum_inv);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        errx(42, "Provide a path to the file with inputs/outputs");
    }

    const float qk_scale = (float)((float)(QK_SCALE / sqrt(128)) * log2f(expf(1.0f)));

    struct InOuts* io = load_file(argv[1], sizeof(struct InOuts));

    #pragma omp parallel for
    for (size_t qh = 0; qh < N_Q_HEADS; ++qh) {
        size_t kvh = qh / (N_Q_HEADS / N_KV_HEADS);
        for (size_t qi = 0; qi < N_Q; ++qi) {
            auto q   = io->q  [qh][qi];
            auto out = OUR_OUT[qh][qi];
            auto ks  = io->k[kvh];
            auto vs  = io->v[kvh];
            compute_query_head(qi, q, out, ks, vs, qk_scale);

            for (size_t ii = 0; ii < HEAD_DIM; ++ii) {
                float ref = to_float(io->out[qh][qi][ii]);
                float our = to_float(out[ii]);
                if (ref != our) {
                    printf(
                        "(%zu, %zu, %zu): %a vs %a\n",
                        qh, qi, ii, ref, our
                    );
                }
            }
        }
    }
}
