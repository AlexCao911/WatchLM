import Foundation
@testable import WatchLMCore

func loadSampleManifest() throws -> ModelManifest {
    let data = Data(sampleManifestJSON.utf8)
    return try JSONDecoder().decode(ModelManifest.self, from: data)
}

func loadStatefulStepCandidateManifest() throws -> ModelManifest {
    let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .appending(path: "tools/validation/fixtures/watch-se2-stateful-step-model-manifest.json")
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode(ModelManifest.self, from: data)
}

func loadQwen3ExplicitKVCandidateManifest() throws -> ModelManifest {
    let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .appending(path: "tools/validation/fixtures/qwen3-0.6b-explicit-kv-model-manifest.json")
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode(ModelManifest.self, from: data)
}

func makeTemporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appending(path: "watchlm-\(UUID().uuidString)", directoryHint: .isDirectory)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

func minimalTokenizerJSONData() -> Data {
    Data(
        """
        {
          "model": {
            "type": "BPE",
            "unk_token": "<unk>",
            "vocab": {
              "<s>": 0,
              "</s>": 1,
              "H": 100,
              "i": 101,
              "Hi": 19301,
              "<unk>": 130074
            },
            "merges": [["H", "i"]]
          },
          "added_tokens": [
            {
              "id": 0,
              "content": "<s>",
              "single_word": false,
              "lstrip": false,
              "rstrip": false,
              "normalized": false,
              "special": true
            },
            {
              "id": 1,
              "content": "</s>",
              "single_word": false,
              "lstrip": false,
              "rstrip": false,
              "normalized": false,
              "special": true
            },
            {
              "id": 130074,
              "content": "<unk>",
              "single_word": false,
              "lstrip": false,
              "rstrip": false,
              "normalized": false,
              "special": true
            }
          ]
        }
        """.utf8
    )
}

private let sampleManifestJSON = """
{
  "schemaVersion": 1,
  "model": {
    "id": "openbmb/MiniCPM5-1B",
    "revision": "design-contract",
    "parameterCount": 1080632832
  },
  "runtime": {
    "type": "coreml-mlprogram",
    "entrypoints": ["prefill", "decode"],
    "kvCacheMode": "stateful-preferred",
    "graphSchema": {
      "interface": "logits-layered-kv",
      "layerCount": 24,
      "kvHeads": 2,
      "headDimension": 128,
      "prefill": {
        "inputIDs": "input_ids",
        "positionIDs": "position_ids",
        "causalMask": "causal_mask",
        "logits": "logits",
        "keyPrefix": "present_key_",
        "valuePrefix": "present_value_"
      },
      "decode": {
        "tokenID": "token_id",
        "positionID": "position_id",
        "causalMask": "causal_mask",
        "logits": "logits",
        "pastKeyPrefix": "past_key_",
        "pastValuePrefix": "past_value_",
        "newKeyPrefix": "new_key_",
        "newValuePrefix": "new_value_"
      }
    }
  },
  "architecture": {
    "type": "LlamaForCausalLM",
    "layers": 24,
    "hiddenSize": 1536,
    "queryHeads": 16,
    "kvHeads": 2,
    "maxContextTokens": 131072,
    "tokenizer": {
      "source": "openbmb/MiniCPM5-1B",
      "preserved": true,
      "vocabularyPreserved": true,
      "chatTemplate": "minicpm5-no-think"
    }
  },
  "deviceProfiles": {
    "watch-se-2": {
      "sip": "S8",
      "neuralEngineCores": 2,
      "defaultContextVariant": 256,
      "maxNewTokens": 64
    },
    "watch-se-3": {
      "sip": "S10",
      "neuralEngineCores": 4,
      "defaultContextVariant": 512,
      "maxNewTokens": 96
    }
  },
  "contextVariants": [256, 512, 1024],
  "asset": {
    "storage": "application-support",
    "prefillPath": "Models/MiniCPM5/prefill-512.mlpackage",
    "decodePath": "Models/MiniCPM5/decode-512.mlpackage",
    "tokenizerPath": "Models/MiniCPM5/tokenizer.json",
    "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "prefillSHA256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
    "decodeSHA256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
    "tokenizerSHA256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    "variants": {
      "256": {
        "deviceProfile": "watch-se-2",
        "prefillPath": "Models/MiniCPM5/prefill-256.mlpackage",
        "decodePath": "Models/MiniCPM5/decode-256.mlpackage",
        "tokenizerPath": "Models/MiniCPM5/tokenizer.json",
        "sha256": "1111111111111111111111111111111111111111111111111111111111111111",
        "prefillSHA256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "decodeSHA256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "tokenizerSHA256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
      },
      "512": {
        "deviceProfile": "watch-se-3",
        "prefillPath": "Models/MiniCPM5/prefill-512.mlpackage",
        "decodePath": "Models/MiniCPM5/decode-512.mlpackage",
        "tokenizerPath": "Models/MiniCPM5/tokenizer.json",
        "sha256": "2222222222222222222222222222222222222222222222222222222222222222",
        "prefillSHA256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
        "decodeSHA256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        "tokenizerSHA256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
      }
    }
  },
  "quantization": {
    "strategy": "mixed-precision-fidelity-first",
    "weights": {
      "embedding": "int8",
      "lmHead": "int8",
      "norms": "fp16",
      "attentionQKO": "int8",
      "attentionV": "int8",
      "ffn": "int4"
    },
    "kvCache": "int8",
    "structuralReduction": false
  },
  "fallbackPolicy": {
    "requiresBenchmarkEvidence": true,
    "order": [
      "reduce-context",
      "adjust-mixed-precision",
      "optimize-lm-head",
      "add-speculative-decoding",
      "vocabulary-pruning-last-resort",
      "layer-pruning-last-resort"
    ]
  }
}
"""
