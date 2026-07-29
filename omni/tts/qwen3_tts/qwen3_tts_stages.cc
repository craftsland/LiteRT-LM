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

#include "omni/tts/qwen3_tts/qwen3_tts_stages.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "support/tokenizer/huggingface_tokenizer.h"  // from @litert
#include "support/util/convert_tensor_buffer.h"  // from @litert
#include "omni/base/io_types.h"
#include "omni/base/stage.h"
#include "omni/tts/acoustic_predictor.h"
#include "omni/tts/latent_decoder.h"
#include "omni/tts/text_frontend.h"
#include "omni/tts/vocoder.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"

namespace litert::omni::tts {
namespace {

constexpr int kTtsBos = 151672;
constexpr int kTtsEos = 151673;
constexpr int kTtsPad = 151671;

constexpr int kCodecVocab = 3072;
constexpr int kCodecPad = 2148;
constexpr int kCodecBos = 2149;
constexpr int kCodecEos = 2150;
constexpr int kCodecThink = 2154;
constexpr int kCodecNoThink = 2155;
constexpr int kCodecThinkBos = 2156;
constexpr int kCodecThinkEos = 2157;

constexpr int kHidden = 1024;
constexpr int kNumCodeGroups = 16;
constexpr float kNegInf = -1e9f;

constexpr const char* kMtpInNames[13] = {
    "args_0", "args_1", "args_2", "args_3",  "args_4",  "args_5", "args_6",
    "args_7", "args_8", "args_9", "args_10", "args_11", "args_12"};

constexpr const char* kMtpOutNames[11] = {
    "output_0", "output_1", "output_2", "output_3", "output_4", "output_5",
    "output_6", "output_7", "output_8", "output_9", "output_10"};

absl::Status CheckFileReadable(const std::string& path) {
  std::ifstream in(path);
  if (!in.good()) {
    return absl::NotFoundError(
        absl::StrCat("File not found or unreadable: ", path));
  }
  return absl::OkStatus();
}

absl::StatusOr<absl::string_view> GetAssetBuffer(
    const Qwen3StageOptions& options, absl::string_view filename) {
  if (options.model_resources != nullptr) {
    auto direct_or = options.model_resources->GetFile(filename);
    if (direct_or.ok()) {
      return direct_or;
    }

    for (absl::string_view entry : options.model_resources->ListFiles()) {
      if (absl::EndsWith(entry, filename)) {
        return options.model_resources->GetFile(entry);
      }
    }
    return absl::NotFoundError(absl::StrCat(
        "Could not find asset '", filename,
        "' in model_resources. Available files: ",
        absl::StrJoin(options.model_resources->ListFiles(), ", ")));
  }
  return absl::InvalidArgumentError("No model_resources provided.");
}

absl::StatusOr<int> GetLanguageId(absl::string_view language) {
  static const auto* const kLanguageMap =
      new absl::flat_hash_map<std::string, int>{
          {"chinese", 2055},  {"english", 2050},    {"german", 2053},
          {"italian", 2070},  {"portuguese", 2071}, {"spanish", 2054},
          {"japanese", 2064}, {"korean", 2064},     {"french", 2061},
          {"russian", 2069},
      };
  std::string lang_lower = std::string(language);
  absl::c_transform(lang_lower, lang_lower.begin(), ::tolower);
  auto it = kLanguageMap->find(lang_lower);
  if (it == kLanguageMap->end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported language: ", language));
  }
  return it->second;
}

absl::StatusOr<CompiledModel> CreateCompiledModel(
    Environment& env, const Qwen3StageOptions& options,
    const std::string& model_filename, int num_threads, bool use_gpu = false) {
  auto comp_options_or = Options::Create();
  if (!comp_options_or.HasValue()) {
    return absl::InternalError("Failed to create Options");
  }
  auto comp_options = std::move(comp_options_or.Value());
  if (use_gpu) {
    LITERT_ASSIGN_OR_RETURN(auto& gpu_compilation_options,
                            comp_options.GetGpuOptions());
    gpu_compilation_options.EnableInfiniteFloatCapping(true);
    gpu_compilation_options.SetPrecision(GpuOptions::Precision::kFp32);
#if defined(__APPLE__)
    gpu_compilation_options.SetPreferTextureWeights(false);
    gpu_compilation_options.SetUseMetalArgumentBuffers(true);
    // gpu_compilation_options.EnableMetalResidencySet(true);
#else   // !__APPLE__
    gpu_compilation_options.SetPreferTextureWeights(true);
#endif  // !__APPLE__
    gpu_compilation_options.EnableConstantTensorSharing(true);
    gpu_compilation_options.SetMadviseOriginalSharedTensors(true);
    gpu_compilation_options.SetConvertWeightsOnGpu(true);
    gpu_compilation_options.SetHintFullyDelegatedToSingleDelegate(true);
    comp_options.SetHardwareAccelerators(HwAccelerators::kGpu);
  } else {
    comp_options.SetHardwareAccelerators(HwAccelerators::kCpu);
    auto cpu_options_or = comp_options.GetCpuOptions();
    if (cpu_options_or.HasValue()) {
      auto& cpu_options = cpu_options_or.Value();
      ABSL_RETURN_IF_ERROR(lm::SetCpuOptions(cpu_options, num_threads));

      if (!options.cache_dir.empty()) {
        std::string cache_path = absl::StrCat(options.cache_dir, "/",
                                              model_filename, ".xnnpack_cache");
        absl::StatusOr<
            std::variant<std::string, std::shared_ptr<lm::ScopedFile>>>
            cache_variant(cache_path);
        ABSL_RETURN_IF_ERROR(
            lm::SetCpuCacheOptions(cache_variant, model_filename, cpu_options));
      }
    }
  }
  if (options.model_resources != nullptr) {
    ABSL_ASSIGN_OR_RETURN(absl::string_view model_buf,
                          GetAssetBuffer(options, model_filename));
    BufferRef<uint8_t> buffer_ref(
        reinterpret_cast<const uint8_t*>(model_buf.data()), model_buf.size());
    LITERT_ASSIGN_OR_RETURN(
        auto compiled_model,
        CompiledModel::Create(env, buffer_ref, comp_options));
    ABSL_VLOG(2) << absl::StrCat("Compiled model created successfully with ",
                                 use_gpu ? "GPU" : "CPU", " backend");
    return std::move(compiled_model);
  }
  std::string path = absl::StrCat(options.model_dir, "/", model_filename);
  ABSL_RETURN_IF_ERROR(CheckFileReadable(path));
  LITERT_ASSIGN_OR_RETURN(auto compiled_model,
                          CompiledModel::Create(env, path, comp_options));
  ABSL_VLOG(2) << absl::StrCat("Compiled model created successfully with ",
                               use_gpu ? "GPU" : "CPU", " backend");
  return std::move(compiled_model);
}

}  // namespace

// --- Stage 1: Qwen3FrontendStage Implementation ---

Qwen3FrontendStage::Qwen3FrontendStage(Stage<std::string>* text_source,
                                       Qwen3StageOptions options,
                                       std::shared_ptr<Environment> env)
    : TextFrontend(text_source),
      options_(std::move(options)),
      env_(std::move(env)) {}

absl::Status Qwen3FrontendStage::Initialize() {
  if (initialized_) return absl::OkStatus();

  if (options_.model_resources != nullptr) {
    ABSL_ASSIGN_OR_RETURN(absl::string_view tok_buf,
                          GetAssetBuffer(options_, options_.tokenizer_file));
    auto tokenizer_or =
        support::HuggingFaceTokenizer::CreateFromJson(std::string(tok_buf));
    if (!tokenizer_or.ok()) {
      return absl::InternalError(
          absl::StrCat("Failed to load tokenizer from buffer: ",
                       tokenizer_or.status().message()));
    }
    tokenizer_ = std::move(*tokenizer_or);
  } else {
    std::string tok_path =
        absl::StrCat(options_.model_dir, "/", options_.tokenizer_file);
    ABSL_RETURN_IF_ERROR(CheckFileReadable(tok_path));
    auto tokenizer_or = support::HuggingFaceTokenizer::CreateFromFile(tok_path);
    if (!tokenizer_or.ok()) {
      return absl::InternalError(absl::StrCat("Failed to load tokenizer from ",
                                              tok_path, ": ",
                                              tokenizer_or.status().message()));
    }
    tokenizer_ = std::move(*tokenizer_or);
  }

  // Load text embedding compiled model if available
  if (env_ != nullptr) {
    auto text_emb_model_or = CreateCompiledModel(
        *env_, options_, options_.text_embedding_file, options_.num_threads);
    if (text_emb_model_or.ok()) {
      text_embedding_model_ = std::move(*text_emb_model_or);
    } else {
      ABSL_LOG(ERROR) << "Failed to load text_embedding_file: "
                      << text_emb_model_or.status();
    }

    // Load text projection compiled model if available
    auto text_proj_model_or = CreateCompiledModel(
        *env_, options_, options_.text_projection_file, options_.num_threads);
    if (text_proj_model_or.ok()) {
      text_projection_model_ = std::move(*text_proj_model_or);
    } else {
      ABSL_LOG(ERROR) << "Failed to load text_projection_file: "
                      << text_proj_model_or.status();
    }

    // Load codec embedding compiled model if available
    auto codec_emb_model_or = CreateCompiledModel(
        *env_, options_, options_.codec_embedding_file, options_.num_threads);
    if (codec_emb_model_or.ok()) {
      codec_embedding_model_ = std::move(*codec_emb_model_or);
    } else {
      ABSL_LOG(ERROR) << "Failed to load codec_embedding_file: "
                      << codec_emb_model_or.status();
    }
  }

  // Load speaker embedding from binary file array
  std::string spk_rel = options_.speaker_file.empty()
                            ? "voices/demo_speaker.bin"
                            : options_.speaker_file;
  if (options_.model_resources != nullptr) {
    auto buf_or = GetAssetBuffer(options_, spk_rel);
    if (buf_or.ok()) {
      absl::string_view buf = *buf_or;
      if (buf.size() == kHidden * sizeof(float)) {
        speaker_emb_.resize(kHidden);
        std::memcpy(speaker_emb_.data(), buf.data(), kHidden * sizeof(float));
      }
    }
  } else {
    std::string path = absl::StrCat(options_.model_dir, "/", spk_rel);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (file.is_open()) {
      std::streamsize size = file.tellg();
      file.seekg(0, std::ios::beg);
      if (size == kHidden * sizeof(float)) {
        speaker_emb_.resize(kHidden);
        file.read(reinterpret_cast<char*>(speaker_emb_.data()), size);
      }
    }
  }

  if (speaker_emb_.size() != kHidden) {
    speaker_emb_.assign(kHidden, 0.0f);
  }

  initialized_ = true;
  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> Qwen3FrontendStage::ProjectText(
    const std::vector<float>& rows, int num_rows) {
  std::vector<float> out(num_rows * 1024);

  LITERT_ASSIGN_OR_RETURN(auto input_buffers,
                          text_projection_model_->CreateInputBuffers());
  LITERT_ASSIGN_OR_RETURN(auto output_buffers,
                          text_projection_model_->CreateOutputBuffers());
  for (int r = 0; r < num_rows; ++r) {
    LITERT_RETURN_IF_ERROR(input_buffers[0].Write<float>(
        absl::MakeConstSpan(rows.data() + r * 2048, 2048)));
    LITERT_RETURN_IF_ERROR(
        text_projection_model_->Run(input_buffers, output_buffers));
    LITERT_RETURN_IF_ERROR(output_buffers[0].Read<float>(
        absl::MakeSpan(out.data() + r * 1024, 1024)));
  }

  return out;
}

absl::StatusOr<std::vector<float>> Qwen3FrontendStage::EmbedCodecToken(
    int code_id) {
  std::vector<float> out(1024, 0.0f);
  if (code_id >= 0 && code_id < 3072 && codec_embedding_model_.has_value()) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffers,
                            codec_embedding_model_->CreateInputBuffers());
    LITERT_ASSIGN_OR_RETURN(auto output_buffers,
                            codec_embedding_model_->CreateOutputBuffers());
    input_buffers[0].Write<int32_t>(absl::MakeConstSpan(&code_id, 1));
    LITERT_RETURN_IF_ERROR(
        codec_embedding_model_->Run(input_buffers, output_buffers));
    LITERT_RETURN_IF_ERROR(
        output_buffers[0].Read<float>(absl::MakeSpan(out.data(), 1024)));
  }
  return out;
}

absl::StatusOr<std::vector<float>> Qwen3FrontendStage::EmbedText(
    const std::vector<int>& ids) {
  int num_ids = ids.size();
  std::vector<float> rows(num_ids * 2048);
  LITERT_ASSIGN_OR_RETURN(auto input_buffers,
                          text_embedding_model_->CreateInputBuffers());
  LITERT_ASSIGN_OR_RETURN(auto output_buffers,
                          text_embedding_model_->CreateOutputBuffers());
  for (int idx = 0; idx < num_ids; ++idx) {
    ABSL_VLOG(2) << "[TRACE] EmbedText: " << ids[idx];
    input_buffers[0].Write<int32_t>(absl::MakeConstSpan(&ids[idx], 1));
    LITERT_RETURN_IF_ERROR(
        text_embedding_model_->Run(input_buffers, output_buffers));
    LITERT_RETURN_IF_ERROR(output_buffers[0].Read<float>(
        absl::MakeSpan(rows.data() + idx * 2048, 2048)));
  }
  return ProjectText(rows, num_ids);
}

absl::StatusOr<FrontendOutput> Qwen3FrontendStage::BuildPrompt(
    const std::string& input_text) {
  std::string prompt_text = absl::StrCat("<|im_start|>assistant\n", input_text,
                                         "<|im_end|>\n<|im_start|>assistant\n");
  auto token_ids_or = tokenizer_->TextToTokenIds(prompt_text);
  if (!token_ids_or.ok()) return token_ids_or.status();
  const std::vector<int>& ids = *token_ids_or;
  if (ids.size() < 8) {
    return absl::InvalidArgumentError(
        "Input text too short for tokenization framing");
  }

  auto sys_embeds_or = EmbedText({kTtsBos, kTtsEos, kTtsPad});
  if (!sys_embeds_or.ok()) return sys_embeds_or.status();
  const std::vector<float>& sys_embeds = *sys_embeds_or;

  std::vector<float> tts_bos(sys_embeds.begin(), sys_embeds.begin() + 1024);
  std::vector<float> tts_eos(sys_embeds.begin() + 1024,
                             sys_embeds.begin() + 2048);
  std::vector<float> tts_pad(sys_embeds.begin() + 2048, sys_embeds.end());

  std::vector<int> control;
  if (options_.language == "auto") {
    control = {kCodecNoThink, kCodecThinkBos, kCodecThinkEos};
  } else {
    auto lang_id_or = GetLanguageId(options_.language);
    if (!lang_id_or.ok()) return lang_id_or.status();
    control = {kCodecThink, kCodecThinkBos, *lang_id_or, kCodecThinkEos};
  }

  std::vector<float> codec_pre;
  auto append_codec_emb = [this, &codec_pre](int code_id) -> absl::Status {
    ABSL_ASSIGN_OR_RETURN(auto vec, EmbedCodecToken(code_id));
    codec_pre.insert(codec_pre.end(), vec.begin(), vec.end());
    return absl::OkStatus();
  };

  for (int c : control) {
    ABSL_RETURN_IF_ERROR(append_codec_emb(c));
  }
  codec_pre.insert(codec_pre.end(), speaker_emb_.begin(), speaker_emb_.end());
  ABSL_RETURN_IF_ERROR(append_codec_emb(kCodecPad));
  ABSL_RETURN_IF_ERROR(append_codec_emb(kCodecBos));

  int n_codec_pre = codec_pre.size() / 1024;

  std::vector<int> role_ids(ids.begin(), ids.begin() + 3);
  auto role_or = EmbedText(role_ids);
  if (!role_or.ok()) return role_or.status();
  std::vector<float> role = *role_or;

  std::vector<float> body((n_codec_pre - 1) * 1024);
  for (int i = 0; i < n_codec_pre - 2; ++i) {
    std::memcpy(body.data() + i * 1024, tts_pad.data(), 1024 * sizeof(float));
  }
  std::memcpy(body.data() + (n_codec_pre - 2) * 1024, tts_bos.data(),
              1024 * sizeof(float));
  for (size_t i = 0; i < body.size(); ++i) {
    body[i] += codec_pre[i];
  }

  auto first_text_emb_or = EmbedText({ids[3]});
  if (!first_text_emb_or.ok()) return first_text_emb_or.status();
  std::vector<float> first_text = *first_text_emb_or;
  const float* codec_pre_last = codec_pre.data() + (n_codec_pre - 1) * 1024;
  for (int i = 0; i < 1024; ++i) {
    first_text[i] += codec_pre_last[i];
  }

  FrontendOutput out;
  out.token_ids = ids;
  out.prompt_embeddings.insert(out.prompt_embeddings.end(), role.begin(),
                               role.end());
  out.prompt_embeddings.insert(out.prompt_embeddings.end(), body.begin(),
                               body.end());
  out.prompt_embeddings.insert(out.prompt_embeddings.end(), first_text.begin(),
                               first_text.end());
  out.prompt_len = out.prompt_embeddings.size() / 1024;

  std::vector<int> trailing_ids;
  if (ids.size() >= 9) {
    trailing_ids.assign(ids.begin() + 4, ids.end() - 5);
  }
  if (!trailing_ids.empty()) {
    auto trailing_or = EmbedText(trailing_ids);
    if (!trailing_or.ok()) return trailing_or.status();
    out.trailing_embeddings = *trailing_or;
  }
  out.trailing_embeddings.insert(out.trailing_embeddings.end(), tts_eos.begin(),
                                 tts_eos.end());
  out.trailing_len = out.trailing_embeddings.size() / 1024;
  out.tts_pad_embedding = std::move(tts_pad);

  return out;
}

absl::Status Qwen3FrontendStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  if (!initialized_) {
    ABSL_RETURN_IF_ERROR(Initialize());
  }
  ABSL_VLOG(2) << "[TRACE] Starting Qwen3FrontendStage::ScheduleInternal";

  auto text_or = text_source_.GetOutput();
  if (absl::IsNotFound(text_or.status())) {
    return absl::OkStatus();
  } else if (!text_or.ok()) {
    return text_or.status();
  }
  ABSL_VLOG(2) << "[TRACE] Running BuildPrompt";
  ABSL_ASSIGN_OR_RETURN(auto prompt, BuildPrompt(*text_or));
  ABSL_VLOG(2) << "[TRACE] Finished BuildPrompt";
  PushOutput(std::move(prompt));
  return absl::OkStatus();
}

void Qwen3FrontendStage::Reset() {
  absl::MutexLock lock(mutex_);
  outputs_.clear();
}

// --- Stage 2: Qwen3AcousticPredictorStage Implementation ---

Qwen3AcousticPredictorStage::Qwen3AcousticPredictorStage(
    Stage<FrontendOutput>* text_frontend, Qwen3StageOptions options,
    std::shared_ptr<Environment> env)
    : AcousticPredictor(text_frontend),
      options_(std::move(options)),
      env_(std::move(env)) {
  if (options_.seed.has_value()) {
    rng_.seed(*options_.seed);
  } else {
    rng_.seed(0);
  }
}

absl::Status Qwen3AcousticPredictorStage::Initialize() {
  if (initialized_) return absl::OkStatus();

  if (!env_) {
    return absl::InvalidArgumentError(
        "Qwen3AcousticPredictorStage requires a non-null LiteRT Environment.");
  }

  ABSL_VLOG(2) << "[TRACE] Creating Talker model";
  ABSL_ASSIGN_OR_RETURN(
      auto talker, CreateCompiledModel(*env_, options_, options_.talker_file,
                                       options_.num_threads,
                                       /*use_gpu=*/false));
  talker_model_ = std::move(talker);
  ABSL_VLOG(2) << "[TRACE] Creating MTP model";
  ABSL_ASSIGN_OR_RETURN(auto mtp,
                        CreateCompiledModel(*env_, options_, options_.mtp_file,
                                            options_.num_threads,
                                            /*use_gpu=*/false));
  mtp_model_ = std::move(mtp);
  for (int i = 0; i < 11; ++i) {
    const char* out_name = kMtpOutNames[i];
    auto out_or = mtp_model_->CreateOutputBuffer("serving_default", out_name);
    if (out_or.HasValue()) {
      mtp_output_map_[out_name] = std::move(out_or.Value());
    } else if (i <= 2) {
      return absl::InternalError(absl::StrCat(
          "Failed to create MTP output buffer in Initialize: ", out_name));
    }
  }

  auto main_in_or = mtp_model_->GetSignatureInputNames("main");
  if (main_in_or.HasValue()) {
    ABSL_VLOG(2) << "[MTP] has signature: main";
  }
  auto serv_in_or = mtp_model_->GetSignatureInputNames("serving_default");
  if (serv_in_or.HasValue()) {
    for (const auto& name : serv_in_or.Value()) {
      ABSL_VLOG(2) << "[MTP] serving_default input: " << name;
    }
  }
  auto serv_out_or = mtp_model_->GetSignatureOutputNames("serving_default");
  if (serv_out_or.HasValue()) {
    for (const auto& name : serv_out_or.Value()) {
      ABSL_VLOG(2) << "[MTP] serving_default output: " << name;
    }
  }

  auto codec_emb_model_or = CreateCompiledModel(
      *env_, options_, options_.codec_embedding_file, options_.num_threads);
  if (codec_emb_model_or.ok()) {
    codec_embedding_model_ = std::move(*codec_emb_model_or);
    LITERT_ASSIGN_OR_RETURN(auto input_buffers,
                            codec_embedding_model_->CreateInputBuffers());
    LITERT_ASSIGN_OR_RETURN(auto output_buffers,
                            codec_embedding_model_->CreateOutputBuffers());
    codec_emb_input_buf_ = std::move(input_buffers[0]);
    codec_emb_output_buf_ = std::move(output_buffers[0]);
  }

  auto mtp_emb_model_or = CreateCompiledModel(
      *env_, options_, options_.mtp_embedding_file, options_.num_threads);
  if (mtp_emb_model_or.ok()) {
    mtp_embedding_model_ = std::move(*mtp_emb_model_or);
    LITERT_ASSIGN_OR_RETURN(auto input_buffers,
                            mtp_embedding_model_->CreateInputBuffers());
    LITERT_ASSIGN_OR_RETURN(auto output_buffers,
                            mtp_embedding_model_->CreateOutputBuffers());
    mtp_emb_input_buf_ = std::move(input_buffers[0]);
    mtp_emb_output_buf_ = std::move(output_buffers[0]);
  }

  auto prefill_inputs_or = talker_model_->GetSignatureInputNames("prefill_32");
  if (prefill_inputs_or.HasValue()) {
    for (auto name : prefill_inputs_or.Value()) {
      ABSL_VLOG(2) << "[TALKER] prefill_32 input: " << name;
    }
  }
  auto prefill_outputs_or =
      talker_model_->GetSignatureOutputNames("prefill_32");
  if (prefill_outputs_or.HasValue()) {
    for (auto name : prefill_outputs_or.Value()) {
      ABSL_VLOG(2) << "[TALKER] prefill_32 output: " << name;
    }
  }

  auto decode_inputs_or = talker_model_->GetSignatureInputNames("decode");
  if (decode_inputs_or.HasValue()) {
    for (auto input_name : decode_inputs_or.Value()) {
      auto buf_or = talker_model_->CreateInputBuffer("decode", input_name);
      if (buf_or.HasValue()) {
        auto sz_or = buf_or.Value().PackedSize();
        size_t sz = sz_or.HasValue() ? sz_or.Value() : 0;
        ABSL_VLOG(2) << "[TALKER-BUF] decode input: " << input_name
                     << " size_bytes=" << sz;
      }
      if (absl::StartsWith(input_name, "kv_cache")) {
        talker_kv_names_.emplace_back(input_name);
      }
    }
  }
  auto decode_outputs_or = talker_model_->GetSignatureOutputNames("decode");
  if (decode_outputs_or.HasValue()) {
    for (auto name : decode_outputs_or.Value()) {
      ABSL_VLOG(2) << "[TALKER] decode output: " << name;
    }
  }

  talker_cache_len_ = 512;
  auto talker_mask_buf_or = talker_model_->CreateInputBuffer("decode", "mask");
  if (talker_mask_buf_or.HasValue()) {
    auto type_or = talker_mask_buf_or.Value().TensorType();
    if (type_or.HasValue()) {
      auto dims = type_or.Value().Layout().Dimensions();
      if (dims.size() > 3) {
        talker_cache_len_ = dims[3];
      }
    }
  }
  (void)talker_model_->CreateInputBuffer("prefill_32", "embeddings");

  initialized_ = true;
  return absl::OkStatus();
}

int Qwen3AcousticPredictorStage::PickToken(const std::vector<float>& logits,
                                           bool do_sample) {
  if (!do_sample) {
    auto max_it = std::max_element(logits.begin(), logits.end());
    return std::distance(logits.begin(), max_it);
  }

  int n = logits.size();
  std::vector<double> scaled(n);
  double temp = std::max(static_cast<double>(options_.temperature), 1e-6);
  double max_logit = -1e30;

  for (int i = 0; i < n; ++i) {
    scaled[i] = static_cast<double>(logits[i]) / temp;
    if (scaled[i] > max_logit) max_logit = scaled[i];
  }

  int top_k = std::min(options_.top_k, n);
  std::vector<std::pair<double, int>> pairs(n);
  for (int i = 0; i < n; ++i) {
    pairs[i] = {scaled[i], i};
  }
  absl::c_partial_sort(
      pairs, pairs.begin() + top_k,
      [](const auto& a, const auto& b) { return a.first > b.first; });

  std::vector<double> probs(top_k);
  double sum_p = 0.0;
  for (int i = 0; i < top_k; ++i) {
    probs[i] = std::exp(pairs[i].first - max_logit);
    sum_p += probs[i];
  }
  for (int i = 0; i < top_k; ++i) {
    probs[i] /= sum_p;
  }

  std::discrete_distribution<int> dist(probs.begin(), probs.end());
  return pairs[dist(rng_)].second;
}

absl::StatusOr<std::vector<float>> Qwen3AcousticPredictorStage::EmbedCodecToken(
    int code_id) {
  std::vector<float> out(kHidden, 0.0f);
  if (codec_embedding_model_.has_value() && codec_emb_input_buf_.has_value() &&
      codec_emb_output_buf_.has_value()) {
    int32_t token = code_id;
    (void)codec_emb_input_buf_->Write<int32_t>(absl::MakeConstSpan(&token, 1));

    auto dup_in_or = codec_emb_input_buf_->Duplicate();
    auto dup_out_or = codec_emb_output_buf_->Duplicate();
    if (dup_in_or.HasValue() && dup_out_or.HasValue()) {
      absl::flat_hash_map<absl::string_view, TensorBuffer> inputs;
      inputs.insert(std::make_pair("codec_ids", std::move(dup_in_or.Value())));
      absl::flat_hash_map<absl::string_view, TensorBuffer> outputs;
      outputs.insert(std::make_pair("output_0", std::move(dup_out_or.Value())));

      if (codec_embedding_model_->Run(inputs, outputs).HasValue()) {
        auto copy_res =
            support::CopyFromTensorBuffer<float>(outputs["output_0"]);
        if (copy_res.HasValue() && copy_res.Value().size() == kHidden) {
          return copy_res.Value();
        }
      }
    }
  }

  if (!codec_emb_.empty()) {
    const float* ptr = codec_emb_.data() + code_id * kHidden;
    std::memcpy(out.data(), ptr, kHidden * sizeof(float));
  }
  return out;
}

absl::StatusOr<std::vector<float>> Qwen3AcousticPredictorStage::EmbedMtpTokens(
    const std::vector<int>& mtp_codes) {
  std::vector<float> sum_embed(kHidden, 0.0f);
  int num_codes = mtp_codes.size();

  if (mtp_embedding_model_.has_value() && mtp_emb_input_buf_.has_value() &&
      mtp_emb_output_buf_.has_value()) {
    for (int g = 0; g < num_codes; ++g) {
      int32_t global_id = g * 2048 + mtp_codes[g];
      (void)mtp_emb_input_buf_->Write<int32_t>(
          absl::MakeConstSpan(&global_id, 1));

      auto dup_in_or = mtp_emb_input_buf_->Duplicate();
      auto dup_out_or = mtp_emb_output_buf_->Duplicate();
      if (!dup_in_or.HasValue() || !dup_out_or.HasValue()) continue;

      absl::flat_hash_map<absl::string_view, TensorBuffer> inputs;
      inputs.insert(std::make_pair("mtp_ids", std::move(dup_in_or.Value())));
      absl::flat_hash_map<absl::string_view, TensorBuffer> outputs;
      outputs.insert(std::make_pair("output_0", std::move(dup_out_or.Value())));

      if (mtp_embedding_model_->Run(inputs, outputs).HasValue()) {
        auto copy_res =
            support::CopyFromTensorBuffer<float>(outputs["output_0"]);
        if (copy_res.HasValue() && copy_res.Value().size() == kHidden) {
          for (int i = 0; i < kHidden; ++i) sum_embed[i] += copy_res.Value()[i];
        }
      }
    }
    return sum_embed;
  }

  if (!mtp_emb_.empty()) {
    for (int g = 0; g < num_codes; ++g) {
      int res_code = mtp_codes[g];
      const float* mtp_ptr = mtp_emb_.data() + (g * 2048 + res_code) * kHidden;
      for (int i = 0; i < kHidden; ++i) sum_embed[i] += mtp_ptr[i];
    }
  }
  return sum_embed;
}

absl::Status Qwen3AcousticPredictorStage::RunPrefill(
    absl::flat_hash_map<absl::string_view, TensorBuffer>& kv_buffers,
    const std::vector<float>& prefill, int p) {
  if (p > 32) {
    return absl::InvalidArgumentError(
        absl::StrCat("Prompt too long for prefill_32: ", p));
  }
  auto emb_buf_or =
      talker_model_->CreateInputBuffer("prefill_32", "embeddings");
  auto pos_buf_or = talker_model_->CreateInputBuffer("prefill_32", "input_pos");
  auto mask_buf_or = talker_model_->CreateInputBuffer("prefill_32", "mask");
  if (!emb_buf_or.HasValue() || !pos_buf_or.HasValue() ||
      !mask_buf_or.HasValue()) {
    return absl::InternalError("Failed to create prefill buffers");
  }
  auto emb_buf = std::move(emb_buf_or.Value());
  auto pos_buf = std::move(pos_buf_or.Value());
  auto mask_buf = std::move(mask_buf_or.Value());

  std::vector<float> emb_32(32 * 1024, 0.0f);
  std::memcpy(emb_32.data(), prefill.data(), p * 1024 * sizeof(float));
  if (!emb_buf.Write<float>(absl::MakeConstSpan(emb_32)).HasValue()) {
    return absl::InternalError("Failed to write prefill emb_buf");
  }

  std::vector<int32_t> pos_32(32);
  for (int i = 0; i < 32; ++i) pos_32[i] = i;
  if (!pos_buf.Write<int32_t>(absl::MakeConstSpan(pos_32)).HasValue()) {
    return absl::InternalError("Failed to write prefill pos_buf");
  }

  std::vector<float> mask_32(32 * talker_cache_len_, kNegInf);
  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j <= i && j < talker_cache_len_; ++j) {
      mask_32[i * talker_cache_len_ + j] = 0.0f;
    }
  }
  if (!mask_buf.Write<float>(absl::MakeConstSpan(mask_32)).HasValue()) {
    return absl::InternalError("Failed to write prefill mask_buf");
  }

  absl::flat_hash_map<absl::string_view, TensorBuffer> input_map;
  input_map["embeddings"] = std::move(emb_buf);
  input_map["input_pos"] = std::move(pos_buf);
  input_map["mask"] = std::move(mask_buf);

  for (const auto& kv_name : talker_kv_names_) {
    if (kv_buffers.contains(kv_name)) {
      input_map[kv_name] = std::move(kv_buffers[kv_name]);
    } else {
      auto kv_buf_or = talker_model_->CreateInputBuffer("prefill_32", kv_name);
      if (kv_buf_or.HasValue()) {
        auto kv_buf = std::move(kv_buf_or.Value());
        auto size_or = kv_buf.PackedSize();
        if (size_or.HasValue() && size_or.Value() > 0) {
          std::vector<float> zeros(size_or.Value() / sizeof(float), 0.0f);
          (void)kv_buf.Write<float>(absl::MakeConstSpan(zeros));
        }
        input_map[kv_name] = std::move(kv_buf);
      }
    }
  }

  absl::flat_hash_map<absl::string_view, TensorBuffer> output_map;
  for (const auto& kv_name : talker_kv_names_) {
    auto out_kv_or = talker_model_->CreateOutputBuffer("prefill_32", kv_name);
    if (out_kv_or.HasValue()) {
      output_map[kv_name] = std::move(out_kv_or.Value());
    }
  }

  auto run_res = talker_model_->Run("prefill_32", input_map, output_map);
  if (!run_res.HasValue()) {
    return absl::InternalError("Talker prefill execution failed");
  }

  for (const auto& kv_name : talker_kv_names_) {
    if (output_map.contains(kv_name)) {
      auto lock_or = TensorBufferScopedLock::Create<const float>(
          output_map[kv_name], TensorBuffer::LockMode::kRead);
      if (lock_or.HasValue() && lock_or.Value().second != nullptr) {
        auto size_or = output_map[kv_name].Size();
        size_t bytes = size_or.HasValue() ? size_or.Value() : 0;
        size_t num_floats = bytes / sizeof(float);

        auto dec_kv_or = talker_model_->CreateInputBuffer("decode", kv_name);
        if (dec_kv_or.HasValue()) {
          auto dec_kv = std::move(dec_kv_or.Value());
          auto dec_size_or = dec_kv.PackedSize();
          size_t total_dec_floats = dec_size_or.HasValue()
                                        ? dec_size_or.Value() / sizeof(float)
                                        : 512 * 8 * 128;
          std::vector<float> dec_buffer(total_dec_floats, 0.0f);
          std::memcpy(dec_buffer.data(), lock_or.Value().second,
                      std::min(num_floats, total_dec_floats) * sizeof(float));
          (void)dec_kv.Write<float>(absl::MakeConstSpan(dec_buffer));
          kv_buffers[kv_name] = std::move(dec_kv);
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<Qwen3AcousticPredictorStage::DecodeOutput>
Qwen3AcousticPredictorStage::RunDecode(
    absl::flat_hash_map<absl::string_view, TensorBuffer>& kv_buffers,
    const float* embed_1024, int pos) {
  auto emb_buf_or = talker_model_->CreateInputBuffer("decode", "embeddings");
  auto pos_buf_or = talker_model_->CreateInputBuffer("decode", "input_pos");
  auto mask_buf_or = talker_model_->CreateInputBuffer("decode", "mask");
  auto logits_buf_or = talker_model_->CreateOutputBuffer("decode", "logits");

  if (!emb_buf_or.HasValue() || !pos_buf_or.HasValue() ||
      !mask_buf_or.HasValue() || !logits_buf_or.HasValue()) {
    return absl::InternalError("Failed to create decode buffers");
  }

  auto emb_buf = std::move(emb_buf_or.Value());
  auto pos_buf = std::move(pos_buf_or.Value());
  auto mask_buf = std::move(mask_buf_or.Value());
  auto logits_buf = std::move(logits_buf_or.Value());

  if (!emb_buf.Write<float>(absl::MakeConstSpan(embed_1024, 1024)).HasValue()) {
    return absl::InternalError("Failed to write decode emb_buf");
  }

  int32_t pos_val = pos;
  if (!pos_buf.Write<int32_t>(absl::MakeConstSpan(&pos_val, 1)).HasValue()) {
    return absl::InternalError("Failed to write decode pos_buf");
  }

  std::vector<float> mask(talker_cache_len_, kNegInf);
  for (int j = 0; j <= pos && j < talker_cache_len_; ++j) {
    mask[j] = 0.0f;
  }
  if (!mask_buf.Write<float>(absl::MakeConstSpan(mask)).HasValue()) {
    return absl::InternalError("Failed to write decode mask_buf");
  }

  absl::flat_hash_map<absl::string_view, TensorBuffer> input_map;
  input_map["embeddings"] = std::move(emb_buf);
  input_map["input_pos"] = std::move(pos_buf);
  input_map["mask"] = std::move(mask_buf);

  for (const auto& kv_name : talker_kv_names_) {
    if (kv_buffers.contains(kv_name)) {
      input_map[kv_name] = std::move(kv_buffers[kv_name]);
    }
  }

  absl::flat_hash_map<absl::string_view, TensorBuffer> output_map;
  output_map["logits"] = std::move(logits_buf);
  for (const auto& kv_name : talker_kv_names_) {
    auto out_kv_or = talker_model_->CreateOutputBuffer("decode", kv_name);
    if (out_kv_or.HasValue()) {
      output_map[kv_name] = std::move(out_kv_or.Value());
    }
  }

  auto run_res = talker_model_->Run("decode", input_map, output_map);
  if (!run_res.HasValue()) {
    return absl::InternalError("Talker decode execution failed");
  }

  for (const auto& kv_name : talker_kv_names_) {
    if (output_map.contains(kv_name)) {
      auto lock_or = TensorBufferScopedLock::Create<const float>(
          output_map[kv_name], TensorBuffer::LockMode::kRead);
      if (lock_or.HasValue() && lock_or.Value().second != nullptr) {
        auto next_in_or = talker_model_->CreateInputBuffer("decode", kv_name);
        if (next_in_or.HasValue()) {
          auto next_in = std::move(next_in_or.Value());
          auto sz_or = output_map[kv_name].Size();
          size_t bytes = sz_or.HasValue() ? sz_or.Value() : 0;
          (void)next_in.Write<float>(absl::MakeConstSpan(
              lock_or.Value().second, bytes / sizeof(float)));
          kv_buffers[kv_name] = std::move(next_in);
        }
      }
    }
  }

  DecodeOutput out;
  out.cb0_logits.resize(kCodecVocab);
  out.hidden.resize(kHidden);

  {
    auto lock_or = TensorBufferScopedLock::Create<const float>(
        output_map["logits"], TensorBuffer::LockMode::kRead);
    if (!lock_or.HasValue()) {
      return absl::InternalError("Failed to lock decode logits output");
    }
    const float* raw_logits = lock_or.Value().second;
    std::memcpy(out.cb0_logits.data(), raw_logits, kCodecVocab * sizeof(float));
    std::memcpy(out.hidden.data(), raw_logits + kCodecVocab,
                kHidden * sizeof(float));
  }

  return out;
}

absl::StatusOr<std::vector<int>> Qwen3AcousticPredictorStage::RunMtp(
    const std::vector<float>& hidden, int cb0) {
  const int num_kv_args = mtp_output_map_.contains(kMtpOutNames[3]) ? 10 : 2;
  const int cache_floats_per_arg = (num_kv_args == 10) ? (8 * 32 * 128) : 87040;
  std::vector<float> zeros(cache_floats_per_arg, 0.0f);

  absl::flat_hash_map<absl::string_view, TensorBuffer> kv_input_buffers;
  for (int i = 0; i < num_kv_args; ++i) {
    const char* arg_name = kMtpInNames[3 + i];
    auto buf_or = mtp_model_->CreateInputBuffer("serving_default", arg_name);
    if (!buf_or.HasValue()) {
      return absl::InternalError(
          absl::StrCat("Failed to create initial MTP input buffer ", arg_name));
    }
    auto buf = std::move(buf_or.Value());
    (void)buf.Write<float>(absl::MakeConstSpan(zeros));
    kv_input_buffers[arg_name] = std::move(buf);
  }

  std::vector<int> codes;
  codes.reserve(15);

  for (int t = 0; t < kNumCodeGroups; ++t) {
    std::vector<float> embed(kHidden, 0.0f);
    if (t == 0) {
      embed = hidden;
    } else if (t == 1) {
      auto emb_or = EmbedCodecToken(cb0);
      if (emb_or.ok()) {
        embed = std::move(*emb_or);
      }
    } else {
      int head_idx = t - 2;
      int prev_code = codes[head_idx];
      if (mtp_embedding_model_.has_value() && mtp_emb_input_buf_.has_value() &&
          mtp_emb_output_buf_.has_value()) {
        int32_t global_id = head_idx * 2048 + prev_code;
        LITERT_RETURN_IF_ERROR(mtp_emb_input_buf_->Write<int32_t>(
            absl::MakeConstSpan(&global_id, 1)));
        LITERT_ASSIGN_OR_RETURN(auto dup_in, mtp_emb_input_buf_->Duplicate());
        LITERT_ASSIGN_OR_RETURN(auto dup_out, mtp_emb_output_buf_->Duplicate());
        std::vector<TensorBuffer> inputs;
        inputs.push_back(std::move(dup_in));
        std::vector<TensorBuffer> outputs;
        outputs.push_back(std::move(dup_out));
        LITERT_RETURN_IF_ERROR(mtp_embedding_model_->Run(inputs, outputs));
        LITERT_RETURN_IF_ERROR(
            outputs[0].Read<float>(absl::MakeSpan(embed.data(), kHidden)));
      } else if (!mtp_emb_.empty()) {
        const float* mtp_ptr =
            mtp_emb_.data() + (head_idx * 2048 + prev_code) * kHidden;
        std::memcpy(embed.data(), mtp_ptr, kHidden * sizeof(float));
      }
    }

    auto arg0_or =
        mtp_model_->CreateInputBuffer("serving_default", kMtpInNames[0]);
    auto arg1_or =
        mtp_model_->CreateInputBuffer("serving_default", kMtpInNames[1]);
    auto arg2_or =
        mtp_model_->CreateInputBuffer("serving_default", kMtpInNames[2]);
    if (!arg0_or.HasValue() || !arg1_or.HasValue() || !arg2_or.HasValue()) {
      return absl::InternalError("Failed to create MTP basic input buffers");
    }
    auto arg0 = std::move(arg0_or.Value());
    auto arg1 = std::move(arg1_or.Value());
    auto arg2 = std::move(arg2_or.Value());

    (void)arg0.Write<float>(absl::MakeConstSpan(embed));
    int32_t t_val = t;
    (void)arg1.Write<int32_t>(absl::MakeConstSpan(&t_val, 1));
    std::vector<float> mask(mtp_cache_len_, kNegInf);
    for (int j = 0; j <= t && j < mtp_cache_len_; ++j) mask[j] = 0.0f;
    (void)arg2.Write<float>(absl::MakeConstSpan(mask));

    absl::flat_hash_map<absl::string_view, TensorBuffer> input_map;
    input_map[kMtpInNames[0]] = std::move(arg0);
    input_map[kMtpInNames[1]] = std::move(arg1);
    input_map[kMtpInNames[2]] = std::move(arg2);
    for (int i = 0; i < num_kv_args; ++i) {
      const char* arg_name = kMtpInNames[3 + i];
      if (kv_input_buffers.contains(arg_name)) {
        input_map[arg_name] = std::move(kv_input_buffers[arg_name]);
      }
    }

    absl::flat_hash_map<absl::string_view, TensorBuffer> step_output_map;
    for (int i = 0; i < 11; ++i) {
      const char* out_name = kMtpOutNames[i];
      auto out_or = mtp_model_->CreateOutputBuffer("serving_default", out_name);
      if (out_or.HasValue()) {
        step_output_map[out_name] = std::move(out_or.Value());
      }
    }

    auto run_res =
        mtp_model_->Run("serving_default", input_map, step_output_map);
    if (!run_res.HasValue()) {
      std::string err_msg = run_res.Error().Message();
      int err_status = static_cast<int>(run_res.Error().Status());
      ABSL_LOG(ERROR) << "[MTP-ERROR] step " << t
                      << " failed: status=" << err_status << " msg=" << err_msg;
      return absl::InternalError(absl::StrCat(
          "MTP step ", t, " failed: ", err_msg, " (status=", err_status, ")"));
    }

    for (int i = 0; i < num_kv_args; ++i) {
      const char* out_name = kMtpOutNames[1 + i];
      const char* next_arg_name = kMtpInNames[3 + i];
      if (step_output_map.contains(out_name)) {
        auto lock_or = TensorBufferScopedLock::Create<const float>(
            step_output_map[out_name], TensorBuffer::LockMode::kRead);
        if (lock_or.HasValue() && lock_or.Value().second != nullptr) {
          auto next_in_or =
              mtp_model_->CreateInputBuffer("serving_default", next_arg_name);
          if (next_in_or.HasValue()) {
            auto next_in = std::move(next_in_or.Value());
            auto sz_or = step_output_map[out_name].Size();
            size_t bytes = sz_or.HasValue()
                               ? sz_or.Value()
                               : (cache_floats_per_arg * sizeof(float));
            (void)next_in.Write<float>(absl::MakeConstSpan(
                lock_or.Value().second, bytes / sizeof(float)));
            kv_input_buffers[next_arg_name] = std::move(next_in);
          }
        }
      }
    }

    if (t >= 1) {
      auto lock_or = TensorBufferScopedLock::Create<const float>(
          step_output_map[kMtpOutNames[0]], TensorBuffer::LockMode::kRead);
      if (!lock_or.HasValue() || lock_or.Value().second == nullptr) {
        return absl::InternalError("Failed to lock MTP output_0 buffer");
      }
      int head_idx = t - 1;
      const float* logits_ptr = lock_or.Value().second + head_idx * 2048;
      std::vector<float> logits(logits_ptr, logits_ptr + 2048);
      int picked_code = PickToken(logits, options_.do_sample);
      codes.push_back(picked_code);
    }
  }

  return codes;
}

absl::Status Qwen3AcousticPredictorStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  if (!initialized_) {
    ABSL_RETURN_IF_ERROR(Initialize());
  }
  ABSL_VLOG(2)
      << "[TRACE] Starting Qwen3AcousticPredictorStage::ScheduleInternal";

  auto frontend_or = text_frontend_.GetOutput();
  if (absl::IsNotFound(frontend_or.status())) {
    return absl::OkStatus();
  } else if (!frontend_or.ok()) {
    return frontend_or.status();
  }
  const auto& frontend = *frontend_or;

  absl::flat_hash_map<absl::string_view, TensorBuffer> kv_buffers;
  for (const auto& kv_name : talker_kv_names_) {
    auto kv_buf_or = talker_model_->CreateInputBuffer("decode", kv_name);
    if (kv_buf_or.HasValue()) {
      auto kv_buf = std::move(kv_buf_or.Value());
      auto size_or = kv_buf.PackedSize();
      if (size_or.HasValue() && size_or.Value() > 0) {
        std::vector<float> zeros(size_or.Value() / sizeof(float), 0.0f);
        (void)kv_buf.Write<float>(absl::MakeConstSpan(zeros));
      }
      kv_buffers[kv_name] = std::move(kv_buf);
    }
  }

  Qwen3AcousticPredictorStage::DecodeOutput decode_out;
  for (int pos = 0; pos < frontend.prompt_len; ++pos) {
    const float* emb_ptr = frontend.prompt_embeddings.data() + pos * 1024;
    ABSL_VLOG(2) << "[TRACE] Running decode for prompt pos=" << pos;
    ABSL_ASSIGN_OR_RETURN(decode_out, RunDecode(kv_buffers, emb_ptr, pos));
  }

  int pos = frontend.prompt_len - 1;

  std::vector<float> suppress(kCodecVocab, 0.0f);
  for (int i = 2048; i < kCodecVocab; ++i) suppress[i] = kNegInf;
  suppress[kCodecEos] = 0.0f;

  std::vector<std::vector<int>> frames;
  std::vector<float> codec_features;
  absl::flat_hash_set<int> history;

  while (static_cast<int>(frames.size()) < options_.max_frames) {
    std::vector<float> scores(kCodecVocab);
    for (int i = 0; i < kCodecVocab; ++i) {
      scores[i] = decode_out.cb0_logits[i] + suppress[i];
    }

    if (frames.size() < 2) {
      scores[kCodecEos] = kNegInf;
    }

    for (int token : history) {
      if (scores[token] > 0) {
        scores[token] /= options_.repetition_penalty;
      } else {
        scores[token] *= options_.repetition_penalty;
      }
    }

    int cb0 = PickToken(scores, options_.do_sample);
    history.insert(cb0);
    ABSL_VLOG(2) << "[TRACE] Generated frame " << frames.size()
                 << " with cb0=" << cb0;
    if (cb0 == kCodecEos) break;

    ABSL_ASSIGN_OR_RETURN(auto mtp_codes, RunMtp(decode_out.hidden, cb0));
    std::vector<int> frame;
    frame.reserve(16);
    frame.push_back(cb0);
    frame.insert(frame.end(), mtp_codes.begin(), mtp_codes.end());
    frames.push_back(std::move(frame));

    ABSL_ASSIGN_OR_RETURN(auto codec_vec, EmbedCodecToken(cb0));
    ABSL_ASSIGN_OR_RETURN(auto mtp_vec, EmbedMtpTokens(mtp_codes));

    std::vector<float> embed(1024);
    for (int i = 0; i < 1024; ++i) {
      float val = codec_vec[i] + mtp_vec[i];
      embed[i] = val;
      codec_features.push_back(val);
    }

    int step = frames.size() - 1;
    if (step < frontend.trailing_len) {
      const float* tr_ptr = frontend.trailing_embeddings.data() + step * 1024;
      for (int i = 0; i < 1024; ++i) embed[i] += tr_ptr[i];
    } else {
      for (int i = 0; i < 1024; ++i) embed[i] += frontend.tts_pad_embedding[i];
    }

    pos += 1;
    if (pos >= talker_cache_len_) {
      ABSL_LOG(WARNING) << "Reached talker_cache_len_ (" << talker_cache_len_
                        << "), stopping decode generation.";
      break;
    }
    ABSL_ASSIGN_OR_RETURN(decode_out, RunDecode(kv_buffers, embed.data(), pos));
  }

  AcousticOutput out;
  out.rvq_frames = std::move(frames);
  out.codec_features = std::move(codec_features);
  PushOutput(std::move(out));
  return absl::OkStatus();
}

void Qwen3AcousticPredictorStage::Reset() {
  absl::MutexLock lock(mutex_);
  outputs_.clear();
}

// --- Stage 3: Qwen3LatentDecoderStage Implementation ---

Qwen3LatentDecoderStage::Qwen3LatentDecoderStage(
    Stage<AcousticOutput>* acoustic_predictor)
    : LatentDecoder(acoustic_predictor) {}

absl::Status Qwen3LatentDecoderStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

  auto acoustic_or = acoustic_predictor_.GetOutput();
  if (absl::IsNotFound(acoustic_or.status())) {
    return absl::OkStatus();
  } else if (!acoustic_or.ok()) {
    return acoustic_or.status();
  }
  const auto& acoustic = *acoustic_or;

  LatentOutput out;
  out.codec_features = acoustic.codec_features;
  out.rvq_frames = acoustic.rvq_frames;
  PushOutput(std::move(out));
  return absl::OkStatus();
}

void Qwen3LatentDecoderStage::Reset() {
  absl::MutexLock lock(mutex_);
  outputs_.clear();
}

// --- Stage 4: Qwen3VocoderStage Implementation ---

Qwen3VocoderStage::Qwen3VocoderStage(Stage<LatentOutput>* latent_decoder,
                                     Qwen3StageOptions options,
                                     std::shared_ptr<Environment> env)
    : Vocoder(latent_decoder),
      options_(std::move(options)),
      env_(std::move(env)) {}

absl::Status Qwen3VocoderStage::Initialize() {
  if (initialized_) return absl::OkStatus();
  ABSL_RETURN_IF_ERROR(EnsureModelInitialized());
  initialized_ = true;
  return absl::OkStatus();
}

absl::Status Qwen3VocoderStage::EnsureModelInitialized() {
  if (codec_model_ != std::nullopt) return absl::OkStatus();

  if (!env_) {
    return absl::InvalidArgumentError(
        "Qwen3VocoderStage requires a non-null LiteRT Environment.");
  }

  ABSL_ASSIGN_OR_RETURN(
      auto codec, CreateCompiledModel(*env_, options_, options_.codec_file,
                                      options_.num_threads));
  codec_model_ = std::move(codec);

  auto input_names_or = codec_model_->GetSignatureInputNames();
  if (input_names_or.HasValue()) {
    for (const auto& name : input_names_or.Value()) {
      ABSL_VLOG(2) << "Codec input name: " << name;
    }
  }
  auto output_names_or = codec_model_->GetSignatureOutputNames();
  if (output_names_or.HasValue()) {
    for (const auto& name : output_names_or.Value()) {
      ABSL_VLOG(2) << "Codec output name: " << name;
    }
  }

  auto codec_buf_or = codec_model_->CreateInputBuffer("args_0");
  if (!codec_buf_or.HasValue()) {
    ABSL_LOG(ERROR) << "Failed to CreateInputBuffer args_0";
  } else {
    auto size_or = codec_buf_or.Value().Size();
    if (size_or.HasValue() && size_or.Value() > 0) {
      codec_chunk_ = size_or.Value() / (sizeof(int32_t) * kNumCodeGroups);
      ABSL_VLOG(2) << "Dynamically set codec_chunk_ = " << codec_chunk_;
    }
  }

  auto codec_out_buf_or = codec_model_->CreateOutputBuffer("output_0");
  if (!codec_out_buf_or.HasValue()) {
    ABSL_LOG(ERROR) << "Failed to CreateOutputBuffer output_0";
  } else if (codec_chunk_ > 0) {
    auto size_or = codec_out_buf_or.Value().Size();
    if (size_or.HasValue() && size_or.Value() > 0) {
      int total_samples = size_or.Value() / sizeof(float);
      upsample_ = total_samples / codec_chunk_;
      ABSL_VLOG(2) << "Dynamically set upsample_ = " << upsample_;
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> Qwen3VocoderStage::DecodeCodes(
    const std::vector<std::vector<int>>& frames) {
  ABSL_VLOG(2) << "[TRACE] Qwen3VocoderStage::DecodeCodes frames.size="
               << frames.size();
  ABSL_RETURN_IF_ERROR(EnsureModelInitialized());
  int num_frames = frames.size();
  if (num_frames == 0) return std::vector<float>{};

  int chunk = codec_chunk_;
  int ctx = 25;

  std::vector<float> waveform;
  int i = 0;

  while (i < num_frames) {
    int c = std::min({ctx, i, chunk - 1});
    int j = std::min(i + chunk - c, num_frames);
    int window_len = j - (i - c);

    auto arg0_or = codec_model_->CreateInputBuffer("args_0");
    auto out0_or = codec_model_->CreateOutputBuffer("output_0");
    if (!arg0_or.HasValue() || !out0_or.HasValue()) {
      return absl::InternalError("Failed to create codec decoder buffers");
    }
    auto arg0 = std::move(arg0_or.Value());
    auto out0 = std::move(out0_or.Value());

    std::vector<int32_t> buf(kNumCodeGroups * chunk, 0);
    for (int k = 0; k < window_len; ++k) {
      int frame_idx = (i - c) + k;
      const auto& frame = frames[frame_idx];
      for (int g = 0; g < kNumCodeGroups && g < static_cast<int>(frame.size());
           ++g) {
        buf[g * chunk + k] = frame[g];
      }
    }
    if (!arg0.Write<int32_t>(absl::MakeConstSpan(buf)).HasValue()) {
      return absl::InternalError("Failed to write codec decoder arg0");
    }

    absl::flat_hash_map<absl::string_view, TensorBuffer> input_map;
    input_map["args_0"] = std::move(arg0);
    absl::flat_hash_map<absl::string_view, TensorBuffer> output_map;
    output_map["output_0"] = std::move(out0);

    auto run_res = codec_model_->Run(input_map, output_map);
    if (!run_res.HasValue()) {
      return absl::InternalError("Codec decoder execution failed");
    }

    auto copy_wav =
        support::CopyFromTensorBuffer<float>(output_map["output_0"]);
    if (!copy_wav.HasValue()) {
      return absl::InternalError("Failed to copy codec decoder output");
    }
    const float* wav_ptr = copy_wav.Value().data();
    size_t total_samples = copy_wav.Value().size();

    int slice_start = c * upsample_;
    int slice_end = window_len * upsample_;
    if (slice_end > static_cast<int>(total_samples)) {
      slice_end = static_cast<int>(total_samples);
    }
    for (int idx = slice_start; idx < slice_end; ++idx) {
      waveform.push_back(wav_ptr[idx]);
    }

    i = j;
  }

  return waveform;
}

absl::Status Qwen3VocoderStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  if (!initialized_) {
    ABSL_RETURN_IF_ERROR(Initialize());
  }

  auto latent_or = latent_decoder_.GetOutput();
  if (absl::IsNotFound(latent_or.status())) {
    return absl::OkStatus();
  } else if (!latent_or.ok()) {
    return latent_or.status();
  }

  pending_frames_.insert(pending_frames_.end(), latent_or->rvq_frames.begin(),
                         latent_or->rvq_frames.end());

  if (static_cast<int>(pending_frames_.size()) >= codec_chunk_) {
    ABSL_ASSIGN_OR_RETURN(auto pcm, DecodeCodes(pending_frames_));
    pending_frames_.clear();
    AudioOutput out;
    out.pcm_samples = std::move(pcm);
    out.sample_rate_hz = 24000;
    PushOutput(std::move(out));
  }
  return absl::OkStatus();
}

absl::Status Qwen3VocoderStage::Flush() {
  while (true) {
    auto latent_or = latent_decoder_.GetOutput();
    if (!latent_or.ok()) break;
    pending_frames_.insert(pending_frames_.end(), latent_or->rvq_frames.begin(),
                           latent_or->rvq_frames.end());
  }

  if (!pending_frames_.empty()) {
    auto pcm_or = DecodeCodes(pending_frames_);
    if (!pcm_or.ok()) {
      ABSL_LOG(ERROR) << "DecodeCodes in Flush failed: " << pcm_or.status();
      return pcm_or.status();
    }
    pending_frames_.clear();
    AudioOutput out;
    out.pcm_samples = std::move(*pcm_or);
    out.sample_rate_hz = 24000;
    PushOutput(std::move(out));
  }
  return absl::OkStatus();
}

void Qwen3VocoderStage::Reset() {
  absl::MutexLock lock(mutex_);
  outputs_.clear();
  pending_frames_.clear();
}

}  // namespace litert::omni::tts
