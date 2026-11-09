# mkdir -p /tmp/jf_dump

# LIBTPU_INIT_ARGS="--xla_jf_dump_to=/tmp/jf_dump \
#   --xla_jf_dump_llo_text=true \
#   --xla_jf_dump_llo_static_gaps=true" \
# python save_eup_results.py 0 16384 exp2

import torch
import sys


FUNCS = {
    'exp2': torch.exp2,
    'log2': torch.log2,
    'rsqrt': torch.rsqrt,
    'recip': torch.reciprocal,
    'sigmoid': torch.sigmoid,
}


def get_inputs(chunk_idx, chunk_size):
    res = torch.arange(
        chunk_idx*chunk_size, (chunk_idx+1)*chunk_size, dtype=torch.int64, device='cpu'
    ).to(torch.uint32).view(torch.float32)
    return res.to(f'tpu')


def get_outputs(chunk_idx, chunk_size, func_name):
    inps = get_inputs(chunk_idx, chunk_size)

    return (
        inps.to('cpu'),
        FUNCS[func_name](inps).to('cpu'),
    )


if __name__ == '__main__':
    chunk_idx = int(sys.argv[1])
    chunk_size = int(sys.argv[2])
    func_name = sys.argv[3]

    torch.save(
        get_outputs(chunk_idx, chunk_size, func_name),
        f'chunk_{chunk_idx}_{chunk_size}_{func_name}.pt'
    )
