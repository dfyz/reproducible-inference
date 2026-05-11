# launch the offline engine
import asyncio

import sglang as sgl
import sglang.test.doc_patch
from sglang.utils import async_stream_and_merge, stream_and_merge
import hashlib

from pathlib import Path


prompts = [
    Path('prompt.txt').read_text(),
]

sampling_params = {
    "temperature": 0.0,
    "max_new_tokens": 65536,
}


def main():
    llm = sgl.Engine(
        model_path="Qwen/Qwen3.5-35B-A3B",
    )

    llm.start_profile()
    outputs = llm.generate(prompts, sampling_params)
    llm.stop_profile()
    generated_text = outputs[0]['text']
    Path('res_sglang.txt').write_text(generated_text)


if __name__ == "__main__":
    main()
