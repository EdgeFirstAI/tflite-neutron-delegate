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

#ifndef HAL_DMABUF_H
#define HAL_DMABUF_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque TFLite delegate pointer */
struct TfLiteDelegate;

/**
 * struct hal_dmabuf_tensor_info - DMA-BUF backing info for a tensor
 * @fd:     DMA-BUF file descriptor (-1 if not backed by dmabuf).
 *          Owned by the delegate -- caller must NOT close it.
 *          Use dup() if a separately-owned fd is needed.
 *          Multiple tensors may share the same fd with different offsets.
 * @offset: Byte offset within the DMA-BUF where tensor data begins.
 *          Use with EGL_DMA_BUF_PLANE0_OFFSET_EXT for EGLImage import.
 *          Always 0 for delegates with one buffer per tensor (VX delegate).
 * @size:   Size in bytes of the tensor's region within the DMA-BUF.
 */
struct hal_dmabuf_tensor_info {
    int fd;
    size_t offset;
    size_t size;
};

/**
 * hal_dmabuf_is_supported - Check if this delegate supports DMA-BUF sharing.
 * @delegate: TFLite delegate pointer.
 *
 * Returns true if the delegate has DMA-BUF buffers available.
 * This is the primary probe function -- if dlsym finds this symbol AND
 * the function returns true, the full hal_dmabuf_* API is usable.
 */
bool hal_dmabuf_is_supported(struct TfLiteDelegate *delegate);

/**
 * hal_dmabuf_get_instance - Get the delegate pointer.
 *
 * Returns the TfLiteDelegate pointer for use with other hal_dmabuf_* calls.
 * Useful when the caller loaded the delegate via TFLite's external delegate
 * loader and needs the pointer for the dmabuf API.
 * Returns NULL if the delegate is not initialized.
 */
struct TfLiteDelegate *hal_dmabuf_get_instance(void);

/**
 * hal_dmabuf_get_tensor_info - Get DMA-BUF backing info for a tensor.
 * @delegate:     TFLite delegate pointer.
 * @tensor_index: TFLite tensor index.
 * @info:         Pointer to caller-allocated struct to fill.
 * @info_size:    sizeof(*info) for ABI safety. The function fills at most
 *                info_size bytes and ignores fields beyond what it knows.
 *
 * Returns 0 on success (info populated), -1 on error (sets errno):
 *   EINVAL  -- delegate is NULL, info is NULL, info_size is 0, or
 *              tensor_index is out of range
 *   ENOENT  -- tensor exists but is not backed by a DMA-BUF
 *   ENOTSUP -- dmabuf not supported or not yet initialized
 */
int hal_dmabuf_get_tensor_info(struct TfLiteDelegate *delegate,
                               int tensor_index,
                               struct hal_dmabuf_tensor_info *info,
                               size_t info_size);

/**
 * hal_dmabuf_sync_for_device - Flush CPU caches before device access.
 * @delegate:     TFLite delegate pointer.
 * @tensor_index: TFLite tensor index.
 *
 * Call after CPU writes to the buffer and before the device (NPU/GPU) reads it.
 *
 * For GPU-to-NPU pipelines, glFinish() ensures GPU writes are complete but
 * does NOT handle CPU cache coherency. If the CPU mmap is cacheable (default),
 * call hal_dmabuf_sync_for_device() after glFinish() and before
 * neutronRunBlocking(). If the buffer is mapped non-cacheable (e.g.,
 * DMA_ATTR_WRITE_COMBINE), glFinish() alone suffices.
 *
 * Returns 0 on success, -1 on error (sets errno).
 */
int hal_dmabuf_sync_for_device(struct TfLiteDelegate *delegate,
                               int tensor_index);

/**
 * hal_dmabuf_sync_for_cpu - Invalidate caches before CPU access.
 * @delegate:     TFLite delegate pointer.
 * @tensor_index: TFLite tensor index.
 *
 * Call before the CPU reads from a buffer that was written by a device.
 * Not needed for H2H pipelines where output goes directly to GPU/display.
 *
 * Returns 0 on success, -1 on error (sets errno).
 */
int hal_dmabuf_sync_for_cpu(struct TfLiteDelegate *delegate,
                            int tensor_index);

#ifdef __cplusplus
}
#endif

#endif /* HAL_DMABUF_H */
