# Copyright 2026 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [Google-internal load of `cc_library`]

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "qwen3_tts_stages",
    srcs = ["qwen3_tts_stages.cc"],
    hdrs = ["qwen3_tts_stages.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_buffer_ref",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc/options:litert_gpu_options",
        "@litert//support/tokenizer:huggingface_tokenizer",
        "@litert//support/util:convert_tensor_buffer",
        "//omni/base:io_types",
        "//omni/base:stage",
        "//omni/tts:acoustic_predictor",
        "//omni/tts:latent_decoder",
        "//omni/tts:text_frontend",
        "//omni/tts:vocoder",
        "//runtime/executor:litert_compiled_model_executor_utils",
        "//runtime/util:memory_mapped_file",
        "//runtime/util:model_asset_bundle_resources",
        "//runtime/util:zip_utils",
    ],
)
