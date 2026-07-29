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

#include "omni/tts/tts_engine.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/base/io_types.h"
#include "omni/tts/qwen3_tts/qwen3_tts_stages.h"
#include "omni/tts/stream_text_source.h"
#include "omni/tts/tts_session.h"
#include "runtime/framework/threadpool.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/model_asset_bundle_resources.h"

namespace litert::omni::tts {

absl::StatusOr<std::unique_ptr<TtsEngine>> TtsEngine::CreateWithComponents(
    const TtsEngineSettings& settings, TtsSession::Components components,
    std::shared_ptr<Environment> env,
    std::shared_ptr<lm::MemoryMappedFile> task_mapped_file) {
  if (components.text_source == nullptr) {
    components.text_source =
        std::make_unique<StreamTextSource>(settings.text_chunk_config);
  }
  StreamTextSource* stream_text_source =
      dynamic_cast<StreamTextSource*>(components.text_source.get());
  if (stream_text_source == nullptr) {
    return absl::InvalidArgumentError(
        "TtsEngine requires components.text_source to be a StreamTextSource.");
  }
  ABSL_ASSIGN_OR_RETURN(auto session,
                        TtsSession::Create(std::move(components)));
  auto thread_pool =
      std::make_unique<lm::ThreadPool>("tts_engine_pool", settings.num_threads);
  if (!env) {
    LITERT_ASSIGN_OR_RETURN(auto shared_env, Environment::Create({}));
    env = std::make_shared<Environment>(std::move(shared_env));
  }
  return std::unique_ptr<TtsEngine>(
      new TtsEngine(settings, std::move(env), std::move(session),
                    stream_text_source, std::move(thread_pool),
                    std::move(task_mapped_file)));
}

absl::StatusOr<std::unique_ptr<TtsEngine>> TtsEngine::Create(
    const TtsEngineSettings& settings) {
  LITERT_ASSIGN_OR_RETURN(auto env, Environment::Create({}));
  auto shared_env = std::make_shared<Environment>(std::move(env));

  TtsSession::Components components;
  components.text_source =
      std::make_unique<StreamTextSource>(settings.text_chunk_config);

  std::shared_ptr<lm::MemoryMappedFile> task_mapped_file = nullptr;

  if (settings.model_type == ModelType::QWEN3_TTS) {
    Qwen3StageOptions options;

    if (settings.model_resources != nullptr) {
      options.model_resources = settings.model_resources;
    } else if (!settings.model_folder.empty() &&
               absl::EndsWith(settings.model_folder, ".task")) {
      auto mapped_file = lm::MemoryMappedFile::Create(settings.model_folder);
      if (!mapped_file.ok()) {
        return absl::InternalError(absl::StrCat(
            "Failed to memory-map task file: ", settings.model_folder, ": ",
            mapped_file.status().message()));
      }
      task_mapped_file = std::move(*mapped_file);
      auto resources =
          lm::ModelAssetBundleResources::Create("", task_mapped_file);
      if (!resources.ok()) {
        return absl::InternalError(absl::StrCat(
            "Failed to parse ModelAssetBundleResources from ",
            settings.model_folder, ": ", resources.status().message()));
      }
      options.model_resources = std::move(*resources);
    }
    options.model_dir = settings.model_folder;
    options.cache_dir = settings.cache_dir;
    options.max_frames = settings.max_frames;

    auto frontend = std::make_unique<Qwen3FrontendStage>(
        components.text_source.get(), options, shared_env);
    auto acoustic = std::make_unique<Qwen3AcousticPredictorStage>(
        frontend.get(), options, shared_env);
    auto latent =
        std::make_unique<Qwen3LatentDecoderStage>(acoustic.get());
    auto vocoder =
        std::make_unique<Qwen3VocoderStage>(latent.get(), options, shared_env);

    ABSL_RETURN_IF_ERROR(frontend->Initialize());
    ABSL_RETURN_IF_ERROR(acoustic->Initialize());
    ABSL_RETURN_IF_ERROR(latent->Initialize());
    ABSL_RETURN_IF_ERROR(vocoder->Initialize());

    components.text_frontend = std::move(frontend);
    components.acoustic_predictor = std::move(acoustic);
    components.latent_decoder = std::move(latent);
    components.vocoder = std::move(vocoder);
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported model_type in TtsEngineSettings: ",
                     static_cast<int>(settings.model_type)));
  }

  return CreateWithComponents(settings, std::move(components),
                              std::move(shared_env),
                              std::move(task_mapped_file));
}

void TtsEngine::Reset() { session_->Reset(); }

absl::StatusOr<AudioOutput> TtsEngine::Flush() {
  stream_text_source_->Finish();
  auto result = session_->Flush();
  if (absl::IsNotFound(result.status())) {
    return AudioOutput();
  }
  return result;
}

absl::StatusOr<AudioOutput> TtsEngine::Synthesize(absl::string_view text) {
  Reset();
  ABSL_RETURN_IF_ERROR(stream_text_source_->PushText(text));
  stream_text_source_->Finish();

  AudioOutput result;
  absl::Notification done;
  absl::Status final_status;
  absl::Mutex mutex;

  absl::Status status = session_->ProcessAsync(
      *thread_pool_,
      [&](absl::StatusOr<AudioOutput> output) -> absl::Status {
        if (absl::IsOutOfRange(output.status())) {
          done.Notify();
          return output.status();
        }
        if (absl::IsNotFound(output.status())) {
          return absl::OkStatus();
        }
        if (!output.ok()) {
          final_status = output.status();
          done.Notify();
          return output.status();
        }
        absl::MutexLock lock(&mutex);
        if (result.sample_rate_hz == 0) {
          result.sample_rate_hz = output->sample_rate_hz;
        }
        result.pcm_samples.insert(result.pcm_samples.end(),
                                  output->pcm_samples.begin(),
                                  output->pcm_samples.end());
        return absl::OkStatus();
      });
  if (!status.ok()) {
    return status;
  }

  done.WaitForNotification();
  if (!final_status.ok()) {
    return final_status;
  }
  return result;
}

absl::Status TtsEngine::SynthesizeAsync(absl::string_view text,
                                        AsyncCallback callback) {
  ABSL_RETURN_IF_ERROR(stream_text_source_->PushText(text));
  absl::Status status =
      session_->ProcessAsync(*thread_pool_, std::move(callback));
  if (absl::IsAlreadyExists(status)) {
    return absl::OkStatus();
  }
  return status;
}

}  // namespace litert::omni::tts
