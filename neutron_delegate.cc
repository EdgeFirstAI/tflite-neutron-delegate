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

#include <utility>
#include <string.h>
#include <vector>
#include <map>
#include <iostream>
#include <fcntl.h>


#include "neutron_delegate.h"

#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/core/subgraph.h"

extern "C" {
#include "neutron/NeutronDriver.h"
}

using namespace std;

namespace tflite {
namespace neutron {

void PrepareNeutronFirmware(TfLiteContext* context) {
    TfLiteTensor* firmware_tensor = NULL;
    for (int i = 0; i < context->tensors_size; i ++){
        auto tensor = &context->tensors[i];
        if (strcmp(tensor->name, "NeutronFirmware") == 0) {
            firmware_tensor = tensor;
        }
    }

    if (firmware_tensor == NULL) {
        system("cp /lib/firmware/NeutronFirmwareDefault.elf /lib/firmware/NeutronFirmware.elf");
    } else {
        int fd = open("/lib/firmware/NeutronFirmware.elf", O_WRONLY | O_CREAT, 0644);
        write(fd, firmware_tensor->data.data, firmware_tensor->bytes);
        close(fd);
    }
}

// Neutron delegate kernel.
class NeutronDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit NeutronDelegateKernel(const NeutronDelegateOptions& opt)
      : options(opt){}

  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    if (options.model_type == NeutronModelType_CONVERTOR) {
      return InitOfflineCompiledModel(context, params);
    } else {
      return InitFineTuningModel(context, params);
    }
  }

  TfLiteStatus InitFineTuningModel(TfLiteContext* context,
                    const TfLiteDelegateParams* params) {
    TF_LITE_ENSURE_EQ(context, params->nodes_to_replace->size, 1);
    operations.resize(1);

    auto &delegate_op = operations[0];
    // Get this node information.
    const int node_index = params->nodes_to_replace->data[0];
    TfLiteNode* node = nullptr;
    TfLiteRegistration* node_registration = nullptr;
    TF_LITE_ENSURE_EQ(
        context,
        context->GetNodeAndRegistration(context, node_index, &node,
                                        &node_registration),
        kTfLiteOk);
    for (int index = 0; index < node->inputs->size - 1; index ++) {
      auto tensor = &context->tensors[node->inputs->data[index]];
      delegate_op.inputs.push_back(node->inputs->data[index]);
      delegate_op.inputs_size.push_back(tensor->bytes);
    }
    for (int index = 0; index < node->outputs->size; index ++) {
      auto tensor = &context->tensors[node->outputs->data[index]];
      delegate_op.outputs.push_back(node->outputs->data[index]);
      delegate_op.outputs_size.push_back(tensor->bytes);
    }
    delegate_op.firmware_input = node->inputs->data[node->inputs->size - 1];

    char *s = getenv("NEUTRON_ENABLE_ZERO_COPY");
    if (s) {
        int val = atoi(s);
        enableZerocp = val == 0 ? false : true;
    }

    return kTfLiteOk;
  }

  TfLiteStatus InitOfflineCompiledModel(TfLiteContext* context,
                    const TfLiteDelegateParams* params) {
    operations.resize(params->nodes_to_replace->size);
    for (int i = 0; i < params->nodes_to_replace->size; ++i) {
      auto &delegate_op = operations[i];
      // Get this node information.
      const int node_index = params->nodes_to_replace->data[i];
      TfLiteNode* node = nullptr;
      TfLiteRegistration* node_registration = nullptr;
      TF_LITE_ENSURE_EQ(
        context,
        context->GetNodeAndRegistration(context, node_index, &node,
                                        &node_registration),
        kTfLiteOk);

      for (int index = 0; index < node->inputs->size; index ++)
        delegate_op.inputs.push_back(node->inputs->data[index]);
      for (int index = 0; index < node->outputs->size; index ++)
        delegate_op.outputs.push_back(node->outputs->data[index]);

      // Get address to microcode data.
      auto mIndex = delegate_op.inputs[node->inputs->size - 3];
      TF_LITE_ENSURE(context, mIndex < context->tensors_size);
      auto mTensor = &context->tensors[mIndex];
      // Set microcode address in neutron structure
      delegate_op.mcfg.microcode = static_cast<const void *>(mTensor->data.raw);

      // Get address to weights data.
      auto wIndex = delegate_op.inputs[node->inputs->size - 2];
      TF_LITE_ENSURE(context, wIndex < context->tensors_size);
      auto wTensor = &context->tensors[wIndex];
      // Set weights address in neutron structure.
      delegate_op.mcfg.weights = static_cast<const void *>(wTensor->data.raw);

      // Get address to kernel data.
      auto kernelIndex = delegate_op.inputs[node->inputs->size - 1];
      TF_LITE_ENSURE(context, kernelIndex < context->tensors_size);
      auto kernelTensor = &context->tensors[kernelIndex];
      // Set kernels address in neutron structure.
      delegate_op.mcfg.kernels = static_cast<const void *>(kernelTensor->data.raw);
    }
    return kTfLiteOk;
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    for (auto& op : operations) {
      if (op.isOpPrepared) {
          continue;
      }
      // Allocate arrays for inputs and outputs
      op.dcfg.inputs = new const void*[op.inputs.size()];
      op.dcfg.outputs = new void*[op.outputs.size()];

      op.isOpPrepared = true;
      if (options.model_type !=NeutronModelType_FFIRMWARE) {
        // Prepare data for through neutron driver.
        auto neutronRC = neutronModelPrepare(&op.mcfg, &op.nmh);
        TF_LITE_ENSURE_EQ(context, neutronRC, ENONE);
      } else if (options.model_type == NeutronModelType_FFIRMWARE) {
        Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
        size_t input_size, output_size;

        TfLiteTensor* firmware_tensor = &context->tensors[op.firmware_input];
        TF_LITE_ENSURE(context, strcmp(firmware_tensor->name, "NeutronFirmware") == 0);
        auto neutronRC = neutronCustomPrepare((uint32_t*)op.inputs_size.data(), op.inputs.size(),
                                              (uint32_t*)op.outputs_size.data(), op.outputs.size(),
                                              firmware_tensor->data.data, firmware_tensor->bytes, &op.nmh);
        TF_LITE_ENSURE_EQ(context, neutronRC, ENONE);
        if (enableZerocp) {
            // Setup input and output tensor ptr to use neutron memory.
            neutronRC = neutronDataSetup(op.nmh, &op.dcfg);
            TF_LITE_ENSURE_EQ(context, neutronRC, ENONE);

            input_size = op.inputs.size();
            output_size = op.outputs.size();

            // alloc for input
            for (int index = 0; index < input_size; index ++) {
                auto tensor_index = op.inputs[index];
                auto tensor = &context->tensors[tensor_index];
                TfLiteCustomAllocation allocation= {(void*)op.dcfg.inputs[index], tensor->bytes};
                this_subgraph->SetCustomAllocationForTensor(tensor_index, allocation, kTfLiteCustomAllocationFlagsSkipAlignCheck);
            }

            // alloc for output
            for (int index = 0; index < output_size; index ++) {
                auto tensor_index = op.outputs[index];
                auto tensor = &context->tensors[tensor_index];
                TfLiteCustomAllocation allocation= {(void*)op.dcfg.outputs[index], tensor->bytes};
                this_subgraph->SetCustomAllocationForTensor(tensor_index, allocation, kTfLiteCustomAllocationFlagsSkipAlignCheck);
            }
        }
      }
    }
    return kTfLiteOk;
  }


  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    for (auto &delegate_op : operations) {
      auto input = &context->tensors[delegate_op.inputs[0]];
      auto output = &context->tensors[delegate_op.outputs[0]];
      if (options.model_type !=NeutronModelType_FFIRMWARE) {
        // Set reference for all inputs
        for (int index = 0; index < delegate_op.inputs.size(); index ++) {
          auto tensor_index = delegate_op.inputs[index];
          auto tensor = &context->tensors[tensor_index];
          delegate_op.dcfg.inputs[index] = tensor->data.raw;
        }

        for (int index = 0; index < delegate_op.outputs.size(); index ++) {
          auto tensor_index = delegate_op.outputs[index];
          auto tensor = &context->tensors[tensor_index];
          delegate_op.dcfg.outputs[index] = tensor->data.raw;
        }
        // Run neutron compute.
        auto neutronRC = neutronRunBlocking(delegate_op.nmh, &delegate_op.dcfg);
        TF_LITE_ENSURE_EQ(context, neutronRC, ENONE);
      } else {
        if (!enableZerocp) {
          // Set reference for all inputs
          for (int index = 0; index < delegate_op.inputs.size(); index ++) {
            auto tensor_index = delegate_op.inputs[index];
            auto tensor = &context->tensors[tensor_index];
            delegate_op.dcfg.inputs[index] = tensor->data.raw;
          }

          for (int index = 0; index < delegate_op.outputs.size(); index ++) {
            auto tensor_index = delegate_op.outputs[index];
            auto tensor = &context->tensors[tensor_index];
            delegate_op.dcfg.outputs[index] = tensor->data.raw;
          }
        }
        auto neutronRC = neutronCustomExec(delegate_op.nmh, &delegate_op.dcfg);
        TF_LITE_ENSURE_EQ(context, neutronRC, ENONE);
      }
    }
    return kTfLiteOk;
  }

  ~NeutronDelegateKernel() {
    for (auto& op : operations) {
      // Unprepare to free resources in neutron driver
      neutronModelUnprepare(op.nmh);
      // Delete arrays for inputs and outputs
      delete[] op.dcfg.inputs;
      delete[] op.dcfg.outputs;
    }
  }

 private:
  struct OperationDataType {
    vector<int> inputs;
    vector<int> outputs;
    vector<int> inputs_size;
    vector<int> outputs_size;

    // Aggregate neutron model and data structures into one
    NeutronModelConfig mcfg;
    NeutronDataConfig dcfg;
    NeutronModelHandle nmh;

    union {
      SliceParams slice;
      ReshapeParams reshape;
      PadParams pad;
    } params;
    int firmware_input;
    bool isOpPrepared = false;
  };
  std::unique_ptr<ModelT> model;

  int slice_input;
  bool enableZerocp = true;

  vector<OperationDataType> operations;
  NeutronDelegateOptions options;
};

// NeutronDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class NeutronDelegate : public SimpleDelegateInterface {
 public:
  explicit NeutronDelegate(const NeutronDelegateOptions& options)
      : options_(options) {
      }
  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    bool ret;
    if (options_.model_type == NeutronModelType_CONVERTOR) {
      ret = (registration->builtin_code == kTfLiteBuiltinCustom &&
             strcmp(registration->custom_name, NEUTRON_CUSTOM_NAME) == 0);
    } else if (options_.model_type == NeutronModelType_FFIRMWARE){
      ret = (registration->builtin_code == kTfLiteBuiltinCustom &&
             strcmp(registration->custom_name, NEUTRON_FIRMWARE_NODE) == 0);
    } else {
      ret = false;
    }
    return ret;
  }

  TfLiteStatus Initialize(TfLiteContext* context) override {
    // Initialize the neutron driver library
    NeutronError err = neutronInit();
    TF_LITE_ENSURE_EQ(context, err, ENONE);

    TfLiteIntArray* plan;
    TfLiteNode* node;
    TfLiteRegistration* registration;
    TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &plan));

    for (int node_index : tflite::TfLiteIntArrayView(plan)) {
      TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
          context, node_index, &node, &registration));
      if (registration->builtin_code == kTfLiteBuiltinCustom &&
          strcmp(registration->custom_name, NEUTRON_CUSTOM_NAME) == 0) {
        options_.model_type = NeutronModelType_CONVERTOR;
	return kTfLiteOk;
      }
      if (registration->builtin_code == kTfLiteBuiltinCustom &&
          strcmp(registration->custom_name, NEUTRON_FIRMWARE_NODE) == 0) {
        options_.model_type = NeutronModelType_FFIRMWARE;
        return kTfLiteOk;
      }
    }

    options_.model_type = NeutronModelType_NORMAL;
    return kTfLiteOk;
  }

  const char* Name() const override {
    static constexpr char kName[] = "NeutronDelegate";
    return kName;
  }

  unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return make_unique<NeutronDelegateKernel>(options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

 private:
  NeutronDelegateOptions options_;
  std::unique_ptr<ModelT> neutron_model;
};

}  // namespace neutron
}  // namespace tflite

NeutronDelegateOptions NeutronDelegateOptionsDefault() {
  NeutronDelegateOptions options;
  options.target = NEUTRON_TARGET;

  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteNeutronDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* NeutronDelegateCreate(const NeutronDelegateOptions* options) {
  auto delegate = make_unique<tflite::neutron::NeutronDelegate>(
          options ? *options : NeutronDelegateOptionsDefault());
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(move(delegate), 
             kTfLiteDelegateFlagsAllowDynamicTensors);
}

// Destroys a delegate created with `NeutronDelegateCreate` call.
void NeutronDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
