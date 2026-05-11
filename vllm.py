from vllm import LLM, SamplingParams
import hashlib
from pathlib import Path

prompts = [
    Path('prompt.txt').read_text(),
]
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=None,
)

def main():
    llm = LLM(
        model="Qwen/Qwen3.5-35B-A3B",
        max_model_len=65536,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": "./vllm_profile",
        }
    )

    llm.start_profile()
    Path('go.txt').touch()
    outputs = llm.generate(prompts, sampling_params)
    llm.stop_profile()

    Path('res_vllm.txt').write_text(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
