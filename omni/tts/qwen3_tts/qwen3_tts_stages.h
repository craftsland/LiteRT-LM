// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_TTS_STAGES_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_TTS_STAGES_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "support/tokenizer/huggingface_tokenizer.h"  // from @litert
#include "omni/base/stage.h"
#include "omni/tts/acoustic_predictor.h"
#include "omni/tts/latent_decoder.h"
#include "omni/tts/text_frontend.h"
#include "omni/tts/vocoder.h"
#include "runtime/util/model_asset_bundle_resources.h"

namespace litert::omni::tts {

struct Qwen3StageOptions {
  std::string model_dir;
  std::shared_ptr<lm::ModelAssetBundleResources> model_resources = nullptr;
  std::string cache_dir;
  // TODO b/538727793 the following fields are fixed for now, will introduce
  // model type specific settings later.
  std::string talker_file = "talker_int4.tflite";
  std::string mtp_file = "mtp_fp32.tflite";
  std::string codec_file = "codec_decoder_fp32.tflite";
  std::string text_embedding_file = "text_embedding.tflite";
  std::string text_projection_file = "text_projection.tflite";
  std::string codec_embedding_file = "codec_embedding.tflite";
  std::string mtp_embedding_file = "mtp_embedding.tflite";
  std::string tokenizer_file = "tokenizer.json";
  std::string speaker_file = "voices/demo_speaker.bin";
  int num_threads = 4;
  int mtp_threads = 2;
  std::string language = "english";
  bool do_sample = true;
  int top_k = 50;
  float temperature = 0.9f;
  float repetition_penalty = 1.05f;
  int max_frames = 512;
  std::optional<uint64_t> seed = 1;
};

// Stage 1: Text Frontend & Prompt Framing for Qwen3-TTS
class Qwen3FrontendStage : public TextFrontend {
 public:
  Qwen3FrontendStage(Stage<std::string>* absl_nonnull text_source,
                     Qwen3StageOptions options,
                     std::shared_ptr<Environment> env = nullptr);
  ~Qwen3FrontendStage() override = default;

  absl::Status Initialize();
  void Reset() override;

 protected:
  absl::Status ScheduleInternal() override;

 private:
  absl::StatusOr<std::vector<float>> EmbedCodecToken(int code_id);
  absl::StatusOr<std::vector<float>> EmbedText(const std::vector<int>& ids);
  absl::StatusOr<std::vector<float>> ProjectText(const std::vector<float>& rows,
                                                 int num_rows);
  absl::StatusOr<FrontendOutput> BuildPrompt(const std::string& input_text);

  Qwen3StageOptions options_;
  std::shared_ptr<Environment> env_;
  std::unique_ptr<support::HuggingFaceTokenizer> tokenizer_;
  std::optional<CompiledModel> text_embedding_model_;
  std::optional<CompiledModel> text_projection_model_;
  std::optional<CompiledModel> codec_embedding_model_;
  std::vector<float> codec_emb_;    // Shape [3072, 1024]
  std::vector<float> speaker_emb_;  // Shape [1024]
  std::vector<float> proj_w1_;      // Shape [2048, 2048]
  std::vector<float> proj_b1_;      // Shape [2048]
  std::vector<float> proj_w2_;      // Shape [1024, 2048]
  std::vector<float> proj_b2_;      // Shape [1024]
  bool initialized_ = false;
};

// Stage 2: Acoustic Predictor (Talker + MTP Autoregressive RVQ Generation)
class Qwen3AcousticPredictorStage : public AcousticPredictor {
 public:
  Qwen3AcousticPredictorStage(Stage<FrontendOutput>* absl_nonnull text_frontend,
                              Qwen3StageOptions options,
                              std::shared_ptr<Environment> env);
  ~Qwen3AcousticPredictorStage() override = default;

  absl::Status Initialize();
  void Reset() override;

 protected:
  absl::Status ScheduleInternal() override;

 private:
  struct DecodeOutput {
    std::vector<float> cb0_logits;  // Shape [3072]
    std::vector<float> hidden;      // Shape [1024]
  };

  absl::Status RunPrefill(
      absl::flat_hash_map<absl::string_view, TensorBuffer>& kv_buffers,
      const std::vector<float>& prefill, int p);
  absl::StatusOr<DecodeOutput> RunDecode(
      absl::flat_hash_map<absl::string_view, TensorBuffer>& kv_buffers,
      const float* embed_1024, int pos);
  absl::StatusOr<std::vector<int>> RunMtp(const std::vector<float>& hidden,
                                          int cb0);
  int PickToken(const std::vector<float>& logits, bool do_sample);
  absl::StatusOr<std::vector<float>> EmbedCodecToken(int code_id);
  absl::StatusOr<std::vector<float>> EmbedMtpTokens(
      const std::vector<int>& mtp_codes);

  Qwen3StageOptions options_;
  std::shared_ptr<Environment> env_;
  std::optional<CompiledModel> talker_model_;
  std::optional<CompiledModel> mtp_model_;
  std::optional<CompiledModel> codec_embedding_model_;
  std::optional<CompiledModel> mtp_embedding_model_;
  std::optional<TensorBuffer> codec_emb_input_buf_;
  std::optional<TensorBuffer> codec_emb_output_buf_;
  std::optional<TensorBuffer> mtp_emb_input_buf_;
  std::optional<TensorBuffer> mtp_emb_output_buf_;
  absl::flat_hash_map<absl::string_view, TensorBuffer> mtp_output_map_;
  std::vector<float> codec_emb_;  // Shape [3072, 1024]
  std::vector<float> mtp_emb_;    // Shape [15, 2048, 1024]
  std::vector<std::string> talker_kv_names_;
  int talker_cache_len_ = 512;
  int mtp_cache_len_ = 32;
  std::mt19937_64 rng_;
  bool initialized_ = false;
};

// Stage 3: Latent Decoder (Discrete RVQ Codebook Index to Feature
// Transformation)
class Qwen3LatentDecoderStage : public LatentDecoder {
 public:
  explicit Qwen3LatentDecoderStage(
      Stage<AcousticOutput>* absl_nonnull acoustic_predictor);
  ~Qwen3LatentDecoderStage() override = default;

  absl::Status Initialize() { return absl::OkStatus(); }
  void Reset() override;

 protected:
  absl::Status ScheduleInternal() override;
};

// Stage 4: Neural Vocoder (Windowed Codec Audio Synthesis)
class Qwen3VocoderStage : public Vocoder {
 public:
  Qwen3VocoderStage(Stage<LatentOutput>* absl_nonnull latent_decoder,
                    Qwen3StageOptions options,
                    std::shared_ptr<Environment> env);
  ~Qwen3VocoderStage() override = default;

  absl::Status Initialize();
  void Reset() override;
  absl::Status Flush() override;

 protected:
  absl::Status ScheduleInternal() override;

 private:
  absl::StatusOr<std::vector<float>> DecodeCodes(
      const std::vector<std::vector<int>>& frames);
  absl::Status EnsureModelInitialized();

  Qwen3StageOptions options_;
  std::shared_ptr<Environment> env_;
  std::optional<CompiledModel> codec_model_;
  int codec_chunk_ = 100;
  int upsample_ = 1920;
  bool initialized_ = false;
  std::vector<std::vector<int>> pending_frames_;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_TTS_STAGES_H_
