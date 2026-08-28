#include "utils/bf16.h"

#include <err.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <stdint.h>
#include <stdio.h>

constexpr size_t DIM              = 6656;
constexpr size_t INTERMEDIATE_DIM = DIM * 3;
constexpr size_t HEAD_DIM         = 128;

constexpr size_t N_Q_HEADS        = 32;
constexpr size_t N_KV_HEADS       = 2;

constexpr size_t VOCAB_SIZE       = 202048;

struct layer {
    bf16 input_ln    [DIM];
    bf16 mlp_down    [DIM]             [INTERMEDIATE_DIM];
    bf16 mlp_gate    [INTERMEDIATE_DIM][DIM];
    bf16 mlp_up      [INTERMEDIATE_DIM][DIM];
    bf16 post_attn_ln[DIM];
    bf16 post_ff_ln  [DIM];
    bf16 pre_ff_ln   [DIM];
    bf16 attn_gate   [N_Q_HEADS] [HEAD_DIM] [DIM];
    bf16 attn_k      [N_KV_HEADS][HEAD_DIM] [DIM];
    bf16 attn_o      [DIM]       [N_Q_HEADS][HEAD_DIM];
    bf16 attn_q      [N_Q_HEADS] [HEAD_DIM] [DIM];
    bf16 attn_v      [N_KV_HEADS][HEAD_DIM] [DIM];
};

struct layers_part_1 {
    bf16 embeds[VOCAB_SIZE][DIM];
    // layers 0 to 44 (inclusive)
    struct layer layers[45];
};

struct layers_part_2 {
    bf16 lm_head[VOCAB_SIZE][DIM];
    // layers 45 to 51 (inclusive)
    struct layer layers[7];
};

struct model {
    struct layers_part_1* p1;
    struct layers_part_2* p2;
};

void* load_st(const char* file_name, void* addr, size_t n_bytes) {
    // We rely on the OS unmapping the file and closing the FD.
    int fd = open(file_name, O_RDONLY);
    if (fd < 0) {
        err(1, "failed to open %s", file_name);
    }

    uint64_t h_len;
    if (read(fd, &h_len, sizeof(h_len)) != sizeof(h_len)) {
        err(2, "failed to read the header length from %s", file_name);
    }

    char* res = mmap(addr, n_bytes, PROT_READ, MAP_SHARED, fd, 0);
    if (res == MAP_FAILED) {
        err(3, "failed to mmap %s", file_name);
    }
    return res + sizeof(h_len) + h_len;
}

struct model load_model() {
    return (struct model) {
        .p1 = load_st("model-00001-of-00002.safetensors", nullptr, sizeof(struct layers_part_1)),
        .p2 = load_st("model-00002-of-00002.safetensors", nullptr, sizeof(struct layers_part_2)),
    };
}

int main() {
    auto model = load_model();

    printf("%a\n", to_float(model.p2->layers[6].attn_gate[11][111][1111]));
}