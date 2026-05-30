# GGUF Q4 route assessment

Date: 2026-05-30

## Question

Should WatchLM use the official GGUF int4/Q4 model as the primary implementation route instead of continuing only with Core ML conversion?

## Current official artifact

OpenBMB publishes `openbmb/MiniCPM5-1B-GGUF` on Hugging Face.

Relevant files in the official repository:

- `MiniCPM5-1B-F16.gguf`: 2.17 GB
- `MiniCPM5-1B-Q8_0.gguf`: 1.15 GB
- `MiniCPM5-1B-Q4_K_M.gguf`: 688 MB

Official usage examples target llama.cpp, llama-cpp-python, Ollama, Docker Model Runner, LM Studio, vLLM, and SGLang.

## Why this is attractive

The Q4_K_M artifact is much closer to an Apple Watch size budget than the current Core ML pair:

- official GGUF Q4_K_M: 688 MB
- current context-256 Core ML prefill+decode pair: about 2.35 GB

GGUF also gives us a known llama.cpp-style runtime contract:

Tokenizer/chat template -> prompt tokens -> prefill -> KV cache -> decode -> sampler.

That contract is exactly the inference chain we are implementing in Swift/Core ML, but llama.cpp already has mature GGUF loaders, quantized matmul kernels, KV cache handling, and sampling logic.

## Why it is not a drop-in replacement on Apple Watch

GGUF is a model container for ggml/llama.cpp-style runtimes. It is not a Core ML graph and cannot be loaded by `MLModel`.

To run it on Apple Watch, we would need a separate native runtime track:

- build a minimal llama.cpp/ggml static library for watchOS
- expose a Swift wrapper for model load, tokenize, prefill, decode, sampling, streaming, and cancellation
- remove or disable unsupported desktop/server features
- validate watchOS memory behavior, thread limits, file loading, and app bundle constraints
- benchmark on SE2/SE3 hardware

The upside is size and mature quantization. The downside is losing Core ML / Neural Engine scheduling and taking on a C/C++ runtime port.

## Recommendation

Add GGUF as a parallel runtime track, not as a replacement until we have device evidence.

Short-term:

- use official Q4_K_M GGUF as a quality and size baseline
- add a `Runtime/GGUF` adapter beside `Runtime/CoreML`
- first target macOS host and watchOS simulator compile
- then test physical SE2/SE3 load, first-token latency, decode tokens/sec, memory, and thermal behavior

Core ML remains useful for Apple-native acceleration, but the current Core ML context-256 route is blocked by prefill quality and artifact size. GGUF is the best practical fallback path to get a real int4 model running sooner.

