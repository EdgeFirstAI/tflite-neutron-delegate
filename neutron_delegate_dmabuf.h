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

/*
 * Register a delegate instance in the dmabuf registry (call from
 * NeutronDelegateCreate). Each delegate carries its own independent
 * tensor→{fd, offset, size} mapping, so multiple interpreter contexts
 * can coexist in one process.
 */
void dmabuf_register(TfLiteDelegate *delegate);

/* Remove one delegate's dmabuf state (call from NeutronDelegateDelete).
 * Other registered delegates are unaffected. */
void dmabuf_unregister(TfLiteDelegate *delegate);

/*
 * Discover DMA-BUF fds for the given tensor virtual addresses of one
 * delegate instance.
 * Call after neutronDataSetup() + SetCustomAllocationForTensor() in Init().
 * Scans /proc/self/fd and /proc/self/maps to correlate tensor vaddrs with
 * neutron dmabuf fds. Populates that delegate's tensor→{fd, offset, size}
 * mapping.
 */
void dmabuf_discover(TfLiteDelegate *delegate,
                     const std::vector<DmabufTensorVaddr> &tensor_vaddrs);

#endif  /* NEUTRON_DELEGATE_DMABUF_H_ */
