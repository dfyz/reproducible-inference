from pathlib import Path
import sys
import torch


def to_bin(tensor):
    return tensor.view(torch.uint8).numpy().tobytes()


def convert(pt_dir, name_prefix, tensor_names):
    in_path = pt_dir / (name_prefix + '.pt')
    out_path = pt_dir / (name_prefix + '.bin')
    pt = torch.load(in_path, map_location='cpu')
    bin_bytes = b''.join(to_bin(pt[name]) for name in tensor_names)
    out_path.write_bytes(bin_bytes)


if __name__ == '__main__':
    pt_dir = Path(sys.argv[1])

    rms_norm_tensors = ['hidden_states_orig', 'hidden_states', 'weight']
    convert(pt_dir, 'pre_norm', rms_norm_tensors)
    convert(pt_dir, 'post_norm', rms_norm_tensors)

    convert(pt_dir, 'qk_norm', ['q_orig', 'k_orig', 'q', 'k'])
    convert(pt_dir, 'rope', ['q_orig', 'k_orig', 'q', 'k'])

    convert(pt_dir, 'attn_gate', ['attn_out_orig', 'attn_out', 'gate'])
    convert(pt_dir, 'act', ['gate_up', 'x'])
