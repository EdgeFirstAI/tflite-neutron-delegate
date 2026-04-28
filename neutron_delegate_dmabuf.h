/*
 * Copyright 2025 Au-Zone Technologies
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

#ifndef NEUTRON_DELEGATE_DMABUF_H_
#define NEUTRON_DELEGATE_DMABUF_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "tensorflow/lite/c/common.h"

/* Tensor virtual address entry for fd discovery correlation */
struct DmabufTensorVaddr {
    int tensor_index;
    uintptr_t vaddr;
    size_t size;
};

/* Set the delegate singleton pointer (call from NeutronDelegateCreate) */
void dmabuf_set_delegate(TfLiteDelegate *delegate);

/* Clear all dmabuf state (call from NeutronDelegateDelete) */
void dmabuf_clear();

/*
 * Discover DMA-BUF fds for the given tensor virtual addresses.
 * Call after neutronDataSetup() + SetCustomAllocationForTensor() in Prepare().
 * Scans /proc/self/fd and /proc/self/maps to correlate tensor vaddrs with
 * neutron dmabuf fds. Populates the global tensor→{fd, offset, size} mapping.
 */
void dmabuf_discover(const std::vector<DmabufTensorVaddr> &tensor_vaddrs);

#endif  /* NEUTRON_DELEGATE_DMABUF_H_ */
