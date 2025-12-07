#!/usr/bin/env python3
"""
Quantize Hermes-4.3-36B to FP8 using llmcompressor (Neural Magic)
Dynamic quantization - no calibration data needed, fast conversion
Output is vLLM-compatible FP8

Requirements:
    pip install llmcompressor torch transformers accelerate

Usage:
    python quantize_fp8.py
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

if __name__ == "__main__":
    print(f"Quantizing {MODEL_PATH} to FP8...")
    print(f"Output: {OUTPUT_PATH}")

    oneshot(
        model=MODEL_PATH,
        output_dir=OUTPUT_PATH,
        recipe=recipe,
        num_calibration_samples=0,
        save_compressed=True,
    )

    print("Done!")
