/*
 * Copyright 2023-2024 NXP
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TENSORFLOW_LITE_DELEGATES_NEUTRON_DELEGATES_H_
#define TENSORFLOW_LITE_DELEGATES_NEUTRON_DELEGATES_H_

#include <memory>
#include <string>

#include "tensorflow/lite/c/common.h"

#define NEUTRON_DELEGATE_VERSION "1.0.0"
#define NEUTRON_TARGET "imx95"
#define NEUTRON_CUSTOM_NAME "NeutronGraph"
#define NEUTRON_FIRMWARE_NODE "NeutronOp"

enum NeutronModelType{
  NeutronModelType_NORMAL = 0,
  NeutronModelType_CONVERTOR,
  NeutronModelType_FFIRMWARE
};

typedef struct {
  std::string target;
  NeutronModelType model_type;
}NeutronDelegateOptions;

// Returns a structure with the default delegate options.
NeutronDelegateOptions NeutronDelegateOptionsDefault();

// Creates a new delegate instance that needs to be destroyed with
// `NeutronDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* NeutronDelegateCreate(const NeutronDelegateOptions* options);

// Destroys a delegate created with `NeutronDelegateCreate` call.
void NeutronDelegateDelete(TfLiteDelegate* delegate);

// A convenient wrapper that returns C++ std::unique_ptr for automatic memory
// management.
inline std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>
NeutronDelegateCreateUnique(const NeutronDelegateOptions* options) {
  return std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      NeutronDelegateCreate(options), NeutronDelegateDelete);
}

#endif  // TENSORFLOW_LITE_DELEGATES_NEUTRON_DELEGATES_H_
