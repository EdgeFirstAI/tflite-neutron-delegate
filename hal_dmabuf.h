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

#ifndef HAL_DMABUF_H_
#define HAL_DMABUF_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void *hal_delegate_t;

typedef enum hal_dtype {
    HAL_DTYPE_U8  = 0,
    HAL_DTYPE_I8  = 1,
    HAL_DTYPE_U16 = 2,
    HAL_DTYPE_I16 = 3,
    HAL_DTYPE_U32 = 4,
    HAL_DTYPE_I32 = 5,
    HAL_DTYPE_U64 = 6,
    HAL_DTYPE_I64 = 7,
    HAL_DTYPE_F16 = 8,
    HAL_DTYPE_F32 = 9,
    HAL_DTYPE_F64 = 10
} hal_dtype;

#define HAL_DMABUF_MAX_NDIM 8

typedef struct hal_dmabuf_tensor_info {
    size_t size;
    size_t offset;
    size_t shape[HAL_DMABUF_MAX_NDIM];
    size_t ndim;
    int fd;
    hal_dtype dtype;
} hal_dmabuf_tensor_info;

typedef struct hal_camera_adaptor_format_info {
    int input_channels;
    int output_channels;
    char fourcc[8];
} hal_camera_adaptor_format_info;

hal_delegate_t hal_dmabuf_get_instance(void);
int hal_dmabuf_is_supported(hal_delegate_t delegate);
int hal_dmabuf_get_tensor_info(hal_delegate_t delegate, int tensor_index,
                               hal_dmabuf_tensor_info *info, size_t info_size);
int hal_dmabuf_sync_for_device(hal_delegate_t delegate, int tensor_index);
int hal_dmabuf_sync_for_cpu(hal_delegate_t delegate, int tensor_index);
int hal_camera_adaptor_is_supported(hal_delegate_t delegate,
                                    const char *format);
int hal_camera_adaptor_get_format_info(hal_delegate_t delegate,
                                       const char *format,
                                       hal_camera_adaptor_format_info *info,
                                       size_t info_size);

#ifdef __cplusplus
}
#endif

#endif  /* HAL_DMABUF_H_ */
