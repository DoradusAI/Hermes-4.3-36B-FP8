---
library_name: transformers
pipeline_tag: text-generation
license: mit
language:
- en
base_model:
- NousResearch/Hermes-4.3-36B
tags:
- hermes
- nous-research
- tool-calling
- hybrid-reasoning
- fp8
- quantized
- vllm
- sglang
---

# Hermes-4.3-36B-FP8

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/68525b342230a897a65cc1c0/kl1EK9zxqt3y2cMm1Cqxc.png" width="55%" alt="Hermes 4.3" />
</div>

## Model Description

This is an **FP8 quantized** version of [NousResearch/Hermes-4.3-36B](https://huggingface.co/NousResearch/Hermes-4.3-36B), created using [llmcompressor](https://github.com/vllm-project/llm-compressor) (Neural Magic).

**Key Benefits:**
- ~47% smaller model size (36GB vs 68GB)
- Native FP8 inference on Ada Lovelace, Hopper, and Blackwell GPUs
- **Single GPU deployment** on 48GB+ cards (RTX 6000 Ada, A100-40GB+)
- Native vLLM and SGLang support with tool calling
- Minimal quality loss with FP8 dynamic quantization

## Key Features

Hermes 4.3 36B is a state-of-the-art instruction-following model with:

- **Hybrid Reasoning**: `<think>...</think>` blocks for deliberative reasoning when needed
- **Native Tool Calling**: Hermes-format tool use with `<tool_call>` tags
- **SOTA RefusalBench**: 74.6% (best among non-abliterated models)
- **Strong Math/Code**: MATH-500 93.8%, competitive AIME performance
- **524K Context**: Extended context window for long documents

## Quantization Details

| Property | Value |
|----------|-------|
| Quantization Method | FP8 Dynamic (W8A8) |
| Weights Precision | FP8 E4M3 (8-bit) |
| Activations Precision | FP8 E4M3 (8-bit, dynamic) |
| Ignored Layers | `lm_head` (kept in BF16) |
| Quantization Tool | llmcompressor 0.12.2 |
| Original Model Size | ~68GB |
| Quantized Model Size | ~36GB |

### Quantization Recipe

```yaml
default_stage:
  default_modifiers:
    QuantizationModifier:
      targets: [Linear]
      ignore: [lm_head]
      scheme: FP8_DYNAMIC
```

## Quick Start with Docker

The easiest way to run this model. No setup required - just Docker with NVIDIA runtime.

### Docker Compose (Recommended)

```bash
# Download docker-compose.yml
wget https://huggingface.co/Doradus/Hermes-4.3-36B-FP8/raw/main/docker/docker-compose.yml

# Run on single GPU (48GB+ recommended)
docker compose up

# Or specify GPU
GPU_ID=0 docker compose up
```

### Docker Run

```bash
# Single GPU (48GB+ VRAM recommended)
docker run --gpus '"device=0"' -p 8000:8000 \
  -v hf_cache:/root/.cache/huggingface \
  --shm-size=16g \
  vllm/vllm-openai:v0.12.0 \
  --model Doradus/Hermes-4.3-36B-FP8 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --tool-call-parser hermes \
  --enable-auto-tool-choice
```

### Test the API

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Doradus/Hermes-4.3-36B-FP8",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Usage

### vLLM (Recommended)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Doradus/Hermes-4.3-36B-FP8 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --trust-remote-code \
  --tool-call-parser hermes \
  --enable-auto-tool-choice
```

### SGLang

```bash
python -m sglang.launch_server \
  --model-path Doradus/Hermes-4.3-36B-FP8 \
  --host 0.0.0.0 \
  --port 8000 \
  --tp 1
```

### Tool Calling Example

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }
    }
}]

response = client.chat.completions.create(
    model="hermes-43-36b-fp8",
    messages=[{"role": "user", "content": "What's the weather like in San Francisco?"}],
    tools=tools,
    tool_choice="auto"
)

print(response.choices[0].message.tool_calls)
# [ToolCall(function=Function(name='get_weather', arguments='{"city": "San Francisco"}'))]
```

## Architecture Details

This is a **dense transformer** model based on ByteDance Seed-OSS architecture:

| Property | Value |
|----------|-------|
| Total Parameters | 36B |
| Hidden Size | 5120 |
| Attention Heads | 80 |
| KV Heads (GQA) | 8 |
| Layers | 64 |
| Intermediate Size | 27648 |
| Max Context | 524,288 tokens |
| Vocabulary | 155,136 tokens |

## Hardware Requirements

### VRAM Analysis

Model weights: **36GB** (vs 68GB BF16 original)

| Context Length | KV Cache (FP16) | Total VRAM | Fits Single GPU? |
|----------------|-----------------|------------|------------------|
| 4K tokens | ~0.5 GB | ~37 GB | A100-40GB (tight) |
| 8K tokens | ~1.0 GB | ~38 GB | A100-40GB |
| 16K tokens | ~2.0 GB | ~39 GB | RTX 6000 Ada (48GB) |
| 32K tokens | ~4.0 GB | ~41 GB | A100-80GB |
| 64K tokens | ~8.0 GB | ~45 GB | A100-80GB / H100 |

*KV cache calculated for GQA with 8 KV heads, 128 head_dim, 64 layers, FP16 KV*

### Recommended Configurations

| GPU Setup | Max Context | Performance | Notes |
|-----------|-------------|-------------|-------|
| 1x RTX 4090 (24GB) | OOM | N/A | Model too large |
| 1x RTX 5090 (32GB) | ~2K tokens | ~5-10 tok/s | Requires `--enforce-eager` |
| **1x RTX 6000 Ada (48GB)** | ~16K tokens | ~20 tok/s | **Recommended single GPU** |
| 1x A100-40GB | ~8K tokens | ~25 tok/s | Single GPU possible |
| **1x A100-80GB** | ~64K tokens | ~40 tok/s | **Recommended production** |
| 1x H100-80GB | ~128K tokens | ~60 tok/s | Full performance |
| 2x RTX 4090 TP=2 | ~16K tokens | ~40 tok/s | Consumer multi-GPU |

**Note**: FP8 inference requires CUDA compute capability 8.9+ (Ada Lovelace) or 9.0+ (Hopper/Blackwell) for optimal performance.

## Quality & Performance

### Original Model Benchmarks (from [NousResearch](https://huggingface.co/NousResearch/Hermes-4.3-36B))

| Benchmark | Hermes 4.3 36B | Description |
|-----------|----------------|-------------|
| **IFEval** | **77.9%** | Instruction following |
| **MMLU-Pro** | **80.7%** | Multi-task understanding |
| **MMLU** | **87.7%** | Multi-task language |
| **RefusalBench** | **74.6%** | Safety/refusal (SOTA) |
| MATH-500 | 93.8% | Mathematical reasoning |
| AIME 24 | 71.9% | Competition math |
| GPQA Diamond | 65.5% | Graduate-level QA |
| BBH | 86.4% | Big-Bench Hard |
| DROP | 83.5% | Reading comprehension |

### FP8 Quantized Benchmarks (lm-evaluation-harness)

| Benchmark | BF16 Original | FP8 Quantized | Degradation |
|-----------|---------------|---------------|-------------|
| **IFEval (prompt-strict)** | 77.9% | **72.46%** | -5.44% |
| IFEval (inst-strict) | - | 80.10% | - |
| IFEval (prompt-loose) | - | 77.08% | - |
| IFEval (inst-loose) | - | 83.81% | - |
| **GSM8K (5-shot strict)** | - | **87.04%** | - |

*Benchmarked 2025-12-07 on RTX PRO 6000 Blackwell (96GB, PCIe Gen5 x16) using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with vLLM 0.12.0, `--apply_chat_template`*

### Measured Throughput

Tested on RTX PRO 6000 Ada (48GB), single GPU, 16K context:

| Test Type | Tokens Generated | Time | Throughput |
|-----------|------------------|------|------------|
| Short reasoning | 100 | 4.61s | **21.7 tok/s** |
| Code generation | 256 | 11.94s | **21.4 tok/s** |
| Long explanation | 512 | 24.42s | **21.0 tok/s** |
| **Average** | 868 | 40.97s | **21.2 tok/s** |

*Tested 2025-12-06 on Doradus infrastructure with vLLM 0.12.0*

**Note**: Current throughput is limited by PCIe x4 bus contention. Expected ~80 tok/s on dedicated PCIe x16 slot.

## Reproduction

To reproduce this quantization:

```python
#!/usr/bin/env python3
"""
Quantize Hermes-4.3-36B to FP8 using llmcompressor (Neural Magic)
Dynamic quantization - no calibration data needed, fast conversion
Output is vLLM-compatible FP8
"""

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
import torch

MODEL_PATH = "NousResearch/Hermes-4.3-36B"
OUTPUT_PATH = "./Hermes-4.3-36B-FP8"

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
)

oneshot(
    model=MODEL_PATH,
    output_dir=OUTPUT_PATH,
    recipe=recipe,
    num_calibration_samples=0,
    save_compressed=True,
)
```

**Requirements:**
```
pip install llmcompressor torch transformers accelerate
```

## Original Model

This quantization is based on [NousResearch/Hermes-4.3-36B](https://huggingface.co/NousResearch/Hermes-4.3-36B).

Hermes 4.3 is NousResearch's first model trained in a **decentralized manner** over the internet using Psyche. Key features:

- Hybrid reasoning mode with `<think>` blocks
- Native Hermes-format tool calling
- SOTA RefusalBench performance (74.6%)
- Strong math, code, and instruction following
- 524K context window

For full details, see the [Hermes 4 Technical Report](https://nousresearch.com/wp-content/uploads/2025/08/Hermes_4_Technical_Report.pdf) (arXiv:2508.18255).

## License

This model inherits the **MIT License** from the original Hermes 4.3 model.

## Citation

If you use this model, please cite the original Hermes 4 paper:

```bibtex
@article{teknium2025hermes4,
  title={Hermes 4 Technical Report},
  author={Teknium, Ryan and Jin, Roger and Suphavadeeprasit, Jai and Mahan, Dakota and Quesnelle, Jeffrey and Li, Joe and Guang, Chen and Sands, Shannon and Malhotra, Karan},
  journal={arXiv preprint arXiv:2508.18255},
  year={2025}
}
```

## Acknowledgements

- [NousResearch](https://nousresearch.com/) for the original Hermes 4.3 model
- [Neural Magic / vLLM](https://github.com/vllm-project/llm-compressor) for llmcompressor
- [DoradusAI](https://doradusonline.com) for the FP8 quantization
