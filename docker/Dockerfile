  # Hermes-4.3-36B-FP8 Inference Server
  # Based on vLLM OpenAI-compatible server
  #
  # Build:
  #   docker build -t hermes-43-36b-fp8 .
  #
  # Run:
  #   docker run --gpus '"device=0"' -p 8000:8000 hermes-43-36b-fp8

  FROM vllm/vllm-openai:v0.12.0

  # Model will be downloaded from HuggingFace on first run
  ENV MODEL_NAME="Doradus/Hermes-4.3-36B-FP8"
  ENV MAX_MODEL_LEN="16384"
  ENV GPU_MEMORY_UTILIZATION="0.90"

  # vLLM settings
  ENV VLLM_ATTENTION_BACKEND="FLASHINFER"

  EXPOSE 8000

  ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]

  CMD ["--model", "Doradus/Hermes-4.3-36B-FP8", \
       "--host", "0.0.0.0", \
       "--port", "8000", \
       "--tensor-parallel-size", "1", \
       "--max-model-len", "16384", \
       "--gpu-memory-utilization", "0.90", \
       "--dtype", "auto", \
       "--trust-remote-code", \
       "--served-model-name", "hermes-43-36b-fp8", \
       "--enable-chunked-prefill", \
       "--max-num-seqs", "16", \
       "--tool-call-parser", "hermes", \
       "--enable-auto-tool-choice"]

