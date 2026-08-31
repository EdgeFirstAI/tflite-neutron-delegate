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
#include <unordered_set>


#include "neutron_delegate.h"
#include "neutron_delegate_dmabuf.h"

#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/core/subgraph.h"
#include "flatbuffers/flexbuffers.h"

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
    char *s = getenv("NEUTRON_ENABLE_ZERO_COPY");
    if (s) {
        int val = atoi(s);
        enableZerocp = val == 0 ? false : true;
    }
    cout << "INFO: Neutron delegate version: v" << NEUTRON_DELEGATE_VERSION
         << "-"
         << GIT_COMMIT_HASH
         << ", "
         << (enableZerocp ? "zerocp enabled." : "non-zerocp.")
         << endl;
    if (options.model_type == NeutronModelType_CONVERTOR) {
      return InitOfflineCompiledModel(context, params);
    } else {
      return InitFineTuningModel(context, params);
    }
  }

  char* NeutronGetAttribute(TfLiteNode* node, const char *key) {
     if (node->custom_initial_data != nullptr && node->custom_initial_data_size > 0) {
        const uint8_t* custom_data = reinterpret_cast<const uint8_t*>(node->custom_initial_data);
        size_t custom_size = static_cast<size_t>(node->custom_initial_data_size);

        // Parse FlexBuffer dictionary
        auto root = flexbuffers::GetRoot(custom_data, custom_size);
        if (root.IsMap()) {
            auto map = root.AsMap();

            if (map[key].IsString()) {
                std::string subgraph_name = map[key].AsString().str();
                char* persistent_name = new char[subgraph_name.length() + 1];
                std::strcpy(persistent_name, subgraph_name.c_str());
		return persistent_name;
            }
        }
      }
      return nullptr;
  }

  TfLiteStatus InitFineTuningModel(TfLiteContext* context,
                    const TfLiteDelegateParams* params) {
    operations.resize(params->nodes_to_replace->size);
    for (int i = 0; i < params->nodes_to_replace->size; ++i) {
      auto &delegate_op = operations[i];
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

      // Get the subgraph name from attribute, like subgraph_030
      delegate_op.mcfg.subgraphName = NeutronGetAttribute(node, "subgraph");

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

        TfLiteTensor* firmware_tensor = &context->tensors[op.firmware_input];
        TF_LITE_ENSURE(context, strcmp(firmware_tensor->name, "NeutronFirmware") == 0);
        auto neutronRC = neutronCustomPrepare((uint32_t*)op.inputs_size.data(), op.inputs.size(),
                                              (uint32_t*)op.outputs_size.data(), op.outputs.size(),
                                              firmware_tensor->data.data, firmware_tensor->bytes, &op.nmh);
        TF_LITE_ENSURE_EQ(context, neutronRC, ENONE);
      }

      if (enableZerocp) {
        Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
        size_t input_size, output_size;

        // Setup input and output tensor ptr to use neutron memory.
        auto neutronRC = neutronDataSetup(op.nmh, &op.dcfg);
        TF_LITE_ENSURE_EQ(context, neutronRC, ENONE);

        input_size = op.inputs.size();
        output_size = op.outputs.size();
        /* Don't set customAllocation for neutron tensors(kernel, microcode, weight, scratch),
           zero-copy for activation only.
         */
        if (options.model_type == NeutronModelType_CONVERTOR) {
          input_size -= 3;
          output_size -= 3; // leave space for profiling and debug tensor
        }
        // Alloc for input tensor
        for (int index = 0; index < input_size; index ++) {
          auto tensor_index = op.inputs[index];
          auto tensor = &context->tensors[tensor_index];

          TfLiteCustomAllocation allocation = {(void*)op.dcfg.inputs[index], tensor->bytes};
          this_subgraph->SetCustomAllocationForTensor(tensor_index, allocation, kTfLiteCustomAllocationFlagsSkipAlignCheck);
        }

        // Alloc for output tensor
        for (int index = 0; index < output_size; index ++) {
          auto tensor_index = op.outputs[index];
          auto tensor = &context->tensors[tensor_index];

          TfLiteCustomAllocation allocation = {(void*)op.dcfg.outputs[index], tensor->bytes};
          this_subgraph->SetCustomAllocationForTensor(tensor_index, allocation, kTfLiteCustomAllocationFlagsSkipAlignCheck);
        }

        // Discover DMA-BUF fds for tensor buffers (hal_dmabuf_* API)
        vector<DmabufTensorVaddr> tensor_vaddrs;
        for (int index = 0; index < input_size; index++) {
          auto tensor_index = op.inputs[index];
          tensor_vaddrs.push_back({tensor_index,
                                   (uintptr_t)op.dcfg.inputs[index],
                                   (size_t)context->tensors[tensor_index].bytes});
        }
        for (int index = 0; index < output_size; index++) {
          auto tensor_index = op.outputs[index];
          tensor_vaddrs.push_back({tensor_index,
                                   (uintptr_t)op.dcfg.outputs[index],
                                   (size_t)context->tensors[tensor_index].bytes});
        }
        dmabuf_discover(params->delegate, tensor_vaddrs);
      }
    }
    return kTfLiteOk;
  }


  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    for (auto &delegate_op : operations) {
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

      if (options.model_type !=NeutronModelType_FFIRMWARE) {
        // Run neutron compute.
        auto neutronRC = neutronRunBlocking(delegate_op.nmh, &delegate_op.dcfg);
        TF_LITE_ENSURE_EQ(context, neutronRC, ENONE);
      } else {
        auto neutronRC = neutronCustomExec(delegate_op.nmh, &delegate_op.dcfg);
        TF_LITE_ENSURE_EQ(context, neutronRC, ENONE);
      }

      // When tensor is shared between two connected neutron buffers, copy it from neutron buffer to tensor->data.raw
      // The next op can use the tensor->data.raw find the correct result
      if (enableZerocp) {
        for (int index = 0; index < delegate_op.outputs.size(); index ++) {
          auto tensor_index = delegate_op.outputs[index];
          auto tensor = &context->tensors[tensor_index];
          auto& v = options.shared_tensors;
          if (std::find(v.begin(), v.end(), tensor_index) != v.end()) {
            memcpy((void *)tensor->data.raw, (void *)delegate_op.dcfg.outputs[index], tensor->bytes);
          }
        }
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


  // Analyzes shared tensors in the entire model.
  // A tensor is "shared" if it is an output of an NPU node and input to another NPU node.
  TfLiteStatus NeutronFindSharedTensors(TfLiteContext* context) {
    // Step 1: Precompute consumer nodes for each tensor
    // tensor_consumers[tensor_idx] = set of NPU nodes consuming tensor_idx
    std::vector<std::unordered_set<int>> tensor_consumers(context->tensors_size);

    TfLiteIntArray* execution_plan;
    context->GetExecutionPlan(context, &execution_plan);

    for (int node_idx=0; node_idx < execution_plan->size; node_idx++) {
        TfLiteNode* node;
        TfLiteRegistration* reg;
        if (context->GetNodeAndRegistration(context, node_idx, &node, &reg) != kTfLiteOk)
            continue;

        bool is_npu_node = (reg->custom_name != nullptr &&
                            std::strstr(reg->custom_name, NEUTRON_CUSTOM_NAME) != nullptr);
        if (!is_npu_node) continue;

        // Record all input tensors consumed by this NPU node
        for (int i = 0; i < node->inputs->size; ++i) {
            int tensor_idx = node->inputs->data[i];
            if (tensor_idx >= 0) {
                tensor_consumers[tensor_idx].insert(node_idx);
            }
        }
    }

    // Step 2: Identify shared tensors
    for (int tensor_idx = 0; tensor_idx < context->tensors_size; ++tensor_idx) {
        // Check if tensor is produced by an NPU node
        bool produced_by_npu = false;
        for (int node_idx = 0; node_idx < execution_plan->size; ++node_idx) {
            TfLiteNode* node;
            TfLiteRegistration* reg;
            context->GetNodeAndRegistration(context, node_idx, &node, &reg);
            bool is_npu_producer = (reg->custom_name != nullptr && 
                                   std::strstr(reg->custom_name, NEUTRON_CUSTOM_NAME) != nullptr);

            // Check if current NPU node outputs the tensor
            for (int j = 0; j < node->outputs->size; ++j) {
                if (node->outputs->data[j] == tensor_idx && is_npu_producer) {
                    produced_by_npu = true;
                    break;
                }
            }
            if (produced_by_npu) break;
        }

        // Shared condition: NPU-produced AND consumed by ≥1 NPU node
        if (produced_by_npu && !tensor_consumers[tensor_idx].empty()) {
            options_.shared_tensors.push_back(tensor_idx);
        }
    }

    return kTfLiteOk;
}

  TfLiteStatus Initialize(TfLiteContext* context) override {
    // Initialize the neutron driver library
    NeutronError err = neutronInit();
    TF_LITE_ENSURE_EQ(context, err, ENONE);

    // Try to find the shared tensors between two neutron nodes
    // Cannot run in NeutronKernelDelegate::init due to the execution plan is different.
    NeutronFindSharedTensors(context);

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
  auto* raw = tflite::TfLiteDelegateFactory::CreateSimpleDelegate(move(delegate),
             kTfLiteDelegateFlagsAllowDynamicTensors);
  dmabuf_register(raw);
  return raw;
}

// Destroys a delegate created with `NeutronDelegateCreate` call.
void NeutronDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
  dmabuf_unregister(delegate);
}
