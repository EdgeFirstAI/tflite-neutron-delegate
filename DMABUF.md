# DMA-BUF Zero-Copy Support for Neutron Delegate

## Document Information

| Field | Value |
|-------|-------|
| **Author** | Sébastien Taylor <sebastien@au-zone.com> |
| **Version** | 0.1.0 |
| **Date** | 2026-03-24 |
| **Status** | Architecture Proposal |

### Changelog

| Version | Date | Description |
|---------|------|-------------|
| 0.1.0 | 2026-03-24 | Initial architecture and requirements |

---

## Overview

This document describes the architecture for adding DMA-BUF zero-copy buffer sharing to the TFLite Neutron Delegate on NXP i.MX95. The goal is to enable GPU↔NPU buffer sharing so that OpenGL preprocessing (running on the Mali GPU) can render directly into NPU input buffers — eliminating CPU memcpy from the inference pipeline.

### Target Use Case

The EdgeFirst AI pipeline performs image preprocessing on the GPU via OpenGL fragment shaders:

```
Camera V4L2 dmabuf fd
  → EGLImage import (Mali GPU)
  → GL texture (sample from)
  → Fragment shader (resize, normalize, color convert)
  → Render into NPU input buffer (dmabuf fd from Neutron delegate)
  → glFinish()
  → neutronRunBlocking() — NPU reads preprocessed data, zero memcpy
  → Output tensor dmabuf fds available for downstream
```

Today this pipeline works on i.MX 8M Plus with the VX Delegate. This document defines how to bring equivalent support to i.MX95 with the Neutron Delegate.

### Why It Doesn't Exist Today

The Neutron kernel driver allocates DMA buffers as anonymous inodes (`anon_inode_getfd`), not proper dmabufs. The Mali GPU cannot import anonymous inodes — it requires `dma_buf_export()`-created file descriptors with standard `dma_buf_ops`. Additionally, the proprietary userspace library (`libNeutronDriver.so`) hides the buffer file descriptors from the delegate, providing only mmap'd virtual pointers.

### What We Proved (Prototype Findings)

A prototype investigation on the imx95-evk confirmed:

1. **The NPU operates on a single contiguous DMA buffer.** All tensors (inputs, outputs, weights, microcode, scratch) are packed at offsets within one buffer allocated via `NEUTRON_IOCTL_BUFFER_CREATE`. The NPU firmware receives offsets from a base DMA address written to hardware registers (`BASEDDRL`, `BASEINOUTL`, `BASESPILLL`).

2. **External dma_heap buffers are invisible to the NPU.** Buffers allocated from `/dev/dma_heap/linux,cma` are not mapped in the NPU's SMMU (iommu group 9 at `490d0000.iommu`). The NPU writes zeros to these buffers — confirmed by on-target testing.

3. **The kernel driver's DMA memory IS compatible with Mali GPU.** The driver uses `dma_alloc_attrs(dev, size, &dma_addr, GFP_KERNEL, DMA_ATTR_FORCE_CONTIGUOUS)` which allocates from the system CMA pool (960 MB on imx95-evk). The Mali Valhall GPU (`4d900000.gpu`) supports `EGL_EXT_image_dma_buf_import` and manages its own MMU page tables — it can map any CMA physical pages.

4. **The kernel driver already creates fds.** `neutron_buffer_create()` calls `anon_inode_getfd("neutron-buffer", ...)` and returns the fd to userspace. The userspace library mmaps it and discards the fd. Converting this to `dma_buf_export()` makes the existing fd a proper dmabuf — no changes to the userspace library required.

5. **Performance is equivalent.** Benchmark on YOLOv8n 640×640: non-zero-copy 36.4 ms, driver zero-copy 35.8 ms, dma_heap zero-copy 36.8 ms (though dma_heap outputs were zeros due to SMMU, the NPU execution path itself ran at full speed).

---

## Architecture

### Driver Stack

```
┌─────────────────────────────────────────────────────┐
│ TFLite Neutron Delegate (neutron_delegate.cc)        │
│   + new: neutron_delegate_dmabuf.cc                  │
│   Exports: hal_dmabuf_* symbols                      │
├─────────────────────────────────────────────────────┤
│ libNeutronDriver.so (NXP proprietary binary)         │
│   neutronDataSetup() → mmap'd pointers               │
│   Buffer fds held internally                         │
│   *** NO CHANGES REQUIRED ***                        │
├─────────────────────────────────────────────────────┤
│ /dev/neutron0 kernel driver (GPL, drivers/staging/)  │
│   neutron_buffer_create() → dma_buf_export() fd      │
│   SMMU-mapped via dma_alloc_attrs()                  │
│   *** PATCH: anon_inode → dma_buf_export ***         │
├─────────────────────────────────────────────────────┤
│ Neutron NPU Hardware (behind SMMU at 490d0000)       │
│   Reads/writes via DMA offsets from base_ddr         │
└─────────────────────────────────────────────────────┘
```

### Buffer Sharing Flow

```
┌──────────┐    ┌────────────┐    ┌──────────────┐    ┌──────────┐
│  Camera  │    │  Mali GPU  │    │  Neutron NPU │    │ Display/ │
│  V4L2    │    │  OpenGL ES │    │  Inference   │    │ Encoder  │
└────┬─────┘    └─────┬──────┘    └──────┬───────┘    └────┬─────┘
     │                │                   │                  │
     │  dmabuf fd     │                   │                  │
     ├───────────────►│                   │                  │
     │                │  EGLImage import  │                  │
     │                │  (camera input)   │                  │
     │                │                   │                  │
     │                │  hal_dmabuf_get_tensor_fd(input)     │
     │                │◄──────────────────┤                  │
     │                │  EGLImage import  │                  │
     │                │  (NPU input buf)  │                  │
     │                │                   │                  │
     │                │  GL render:       │                  │
     │                │  sample camera    │                  │
     │                │  write to NPU buf │                  │
     │                │                   │                  │
     │                │  glFinish()       │                  │
     │                ├──────────────────►│                  │
     │                │                   │  NPU inference   │
     │                │                   │  (reads input    │
     │                │                   │   from dmabuf)   │
     │                │                   │                  │
     │                │                   │  hal_dmabuf_get_tensor_fd(output)
     │                │                   ├─────────────────►│
     │                │                   │  dmabuf fd       │
     │                │                   │  (zero-copy out) │
```

### What Changes Where

| Component | Repository | Change | Delivery |
|-----------|-----------|--------|----------|
| Kernel driver | `nxp-imx/linux-imx` | Patch `neutron_buffer.c`: `anon_inode_getfd` → `dma_buf_export` | `meta-edgefirst` bbappend |
| Neutron delegate | `EdgeFirstAI/tflite-neutron-delegate` (fork) | Add `neutron_delegate_dmabuf.cc`, export `hal_dmabuf_*` symbols | Fork with feature branch |
| edgefirst-tflite | `EdgeFirst/tflite-rs` | Update `try_load()` to probe `hal_dmabuf_*` symbols | Feature branch |
| EdgeFirst HAL | `EdgeFirst/hal` | Formalize `hal_dmabuf_*` interface in C header | Feature branch |
| VX delegate | `EdgeFirstAI/tflite-vx-delegate-imx` (fork) | Add `hal_dmabuf_*` symbol exports (thin wrappers around existing `VxDelegate*` functions) | Feature branch |

---

## Kernel Driver Patch

### Current Code (`neutron_buffer.c:78-123`)

The existing `neutron_buffer_create()` allocates DMA memory and returns an anonymous inode fd:

```c
buf->cpu_addr = dma_alloc_attrs(buf->ndev->dev, size,
                &buf->dma_addr, GFP_KERNEL, DMA_ATTR_FORCE_CONTIGUOUS);

ret = anon_inode_getfd("neutron-buffer", &neutron_buffer_fops, buf,
                       O_RDWR | O_CLOEXEC);
```

### Required Change

Replace the anonymous inode with a proper `dma_buf_export()`. **No changes to `libNeutronDriver.so` are required.** The proprietary userspace library only performs `mmap()` and `close()` on the buffer fd — both work identically with dma_buf fds.

#### Userspace Compatibility (`libNeutronDriver.so` — unchanged)

| Userspace operation | Current (anon_inode) | After patch (dma_buf) | Compatible? |
|---|---|---|---|
| `mmap(fd)` | `neutron_buffer_fops.mmap` → `dma_mmap_attrs()` | `dma_buf_fops.mmap` → our `dma_buf_ops.mmap` → same `dma_mmap_attrs()` | **Yes** |
| `close(fd)` | `neutron_buffer_fops.release` → `neutron_buffer_put()` | `dma_buf_fops.release` → our `dma_buf_ops.release` → same `neutron_buffer_put()` | **Yes** |
| `ioctl(fd, ...)` | No ioctl handler on buffer fd | dma_buf framework adds `DMA_BUF_IOCTL_SYNC` automatically | **Yes** (bonus) |

The buffer fd's observable behavior from userspace is identical. The library receives an fd, mmaps it, and uses the virtual pointer — this path is unchanged.

#### Kernel Changes (GPL code only)

**1. Implement `struct dma_buf_ops`** for the neutron buffer:

- `map_dma_buf` — Returns a single-entry `sg_table` for the CMA allocation (`DMA_ATTR_FORCE_CONTIGUOUS` guarantees physical contiguity). Must call `dma_map_sgtable()` on the **importing** device (e.g., Mali GPU), not the neutron device. This is the callback that enables other devices to create their own IOMMU mappings via `dma_buf_attach()` + `dma_buf_map_attachment()`.
- `unmap_dma_buf` — Calls `dma_unmap_sgtable()` and frees the `sg_table`.
- `mmap` — Reuses the existing `dma_mmap_attrs()` call from `neutron_buffer_mmap()`.
- `release` — Calls `neutron_buffer_put()` (same as current `neutron_buffer_release()`).
- `begin_cpu_access` / `end_cpu_access` — Delegate to `neutron_memory_sync()` for cache maintenance. This ensures `DMA_BUF_IOCTL_SYNC` works correctly and is consistent with the existing `NEUTRON_IOCTL_CACHE_SYNC` path.

Reference `map_dma_buf` implementation for a CMA-backed contiguous buffer:

```c
static struct sg_table *neutron_map_dma_buf(struct dma_buf_attachment *attach,
                                            enum dma_data_direction dir)
{
    struct neutron_buffer *buf = attach->dmabuf->priv;
    struct sg_table *sgt;
    int ret;

    sgt = kzalloc(sizeof(*sgt), GFP_KERNEL);
    if (!sgt)
        return ERR_PTR(-ENOMEM);
    ret = sg_alloc_table(sgt, 1, GFP_KERNEL);
    if (ret) {
        kfree(sgt);
        return ERR_PTR(ret);
    }
    sg_set_page(sgt->sgl,
                phys_to_page(dma_to_phys(buf->ndev->dev, buf->dma_addr)),
                buf->size, 0);
    ret = dma_map_sgtable(attach->dev, sgt, dir, 0);
    if (ret) {
        sg_free_table(sgt);
        kfree(sgt);
        return ERR_PTR(ret);
    }
    return sgt;
}
```

**2. In `neutron_buffer_create()`**, replace:

```c
/* Before */
ret = anon_inode_getfd("neutron-buffer", &neutron_buffer_fops, buf,
                       O_RDWR | O_CLOEXEC);

/* After */
DEFINE_DMA_BUF_EXPORT_INFO(exp_info);
exp_info.exp_name = "neutron";    /* identifies exporter in /proc and debugfs */
exp_info.ops = &neutron_dmabuf_ops;
exp_info.size = size;
exp_info.priv = buf;              /* recover via dmabuf->priv */
exp_info.flags = O_RDWR | O_CLOEXEC;
buf->dmabuf = dma_buf_export(&exp_info);
if (IS_ERR(buf->dmabuf))
    goto free_dma;
ret = dma_buf_fd(buf->dmabuf, O_RDWR | O_CLOEXEC);
if (ret < 0)
    goto free_dmabuf;
```

**3. In `neutron_buffer_get_from_fd()`**, change the fd→struct recovery path:

```c
/* Before: anon_inode stores our struct in file->private_data */
struct neutron_buffer *neutron_buffer_get_from_fd(int fd)
{
    struct file *file = fget(fd);
    if (!file)
        return ERR_PTR(-EINVAL);
    buf = file->private_data;
    fput(file);
    return buf;
}

/* After: dma_buf stores our struct in dmabuf->priv.
 * CRITICAL: validate that the fd is a neutron-exported dmabuf before
 * dereferencing priv. Without this check, passing an fd from a different
 * exporter (DRM GEM, V4L2, dma_heap) causes kernel memory corruption. */
struct neutron_buffer *neutron_buffer_get_from_fd(int fd)
{
    struct dma_buf *dmabuf = dma_buf_get(fd);
    if (IS_ERR(dmabuf))
        return ERR_PTR(-EINVAL);
    if (dmabuf->ops != &neutron_dmabuf_ops) {
        dma_buf_put(dmabuf);
        return ERR_PTR(-EINVAL);
    }
    struct neutron_buffer *buf = dmabuf->priv;
    dma_buf_put(dmabuf);
    return buf;
}
```

This function is called from 3 kernel-internal sites (all GPL code):

| Call site | Purpose |
|---|---|
| `neutron_inference.c:469` | `NEUTRON_IOCTL_INFERENCE_CREATE` — recover buffer for inference job |
| `neutron_device.c:561` | `NEUTRON_IOCTL_CACHE_SYNC` — recover buffer for cache maintenance |
| `neutron_device.c:602` | `NEUTRON_IOCTL_FIRMWARE_LOAD` — recover buffer for firmware loading |

All three are purely kernel-internal and invisible to userspace. No ABI change.

#### What This Enables

Once the buffer fd is a proper dmabuf:
- **Mali GPU** can import it via `dma_buf_attach()` → EGLImage → GL texture/renderbuffer
- **V4L2 / DRM / other devices** can import it via standard dmabuf sharing
- **DMA_BUF_IOCTL_SYNC** works on the buffer fd for cache coherency (free bonus from the dma_buf framework)
- The delegate can discover the fd via `/proc/self/fd` and expose it through the `hal_dmabuf_*` API

### Delivery

The patch is delivered as a `.bbappend` file in the `meta-edgefirst` Yocto layer (`github.com/EdgeFirstAI/meta-edgefirst`). This keeps the NXP kernel source unmodified while layering our change on top.

### First Test: Validate fd Visibility

Before any delegate work begins, the kernel patch must be deployed and the following validated on target:

**Does `libNeutronDriver.so` keep the buffer fd open after mmap?** If the library calls `close(fd)` after `mmap()`, the fd disappears from `/proc/self/fd` and the entire fd discovery mechanism fails. The mmap itself survives (the kernel holds a reference), but the delegate has no way to find the fd.

This is a go/no-go gate for the delegate's fd discovery approach. Test immediately after deploying the kernel patch:

```bash
# Run benchmark in background, inspect its open fds
NEUTRON_ENABLE_ZERO_COPY=1 benchmark_model \
    --external_delegate_path=libneutron_delegate.so \
    --graph=model.tflite --num_runs=1000 &
PID=$!
sleep 2

# Look for neutron dmabuf fds (after kernel patch, readlink shows /dmabuf:neutron)
ls -la /proc/$PID/fd/ | grep dmabuf
# Or check all fds:
for fd in /proc/$PID/fd/*; do
    target=$(readlink $fd 2>/dev/null)
    echo "fd=$(basename $fd) -> $target"
done | grep -i neutron

kill $PID
```

**If neutron dmabuf fds are visible**: Approach A (fd scanning) works. Proceed with delegate implementation.

**If no neutron fds are found**: The library closes the fd after mmap. Fallback options:
1. Add `NEUTRON_IOCTL_GET_BUFFER_FD` to the kernel patch — the delegate calls this ioctl on `/dev/neutron0` to retrieve the dmabuf fd for a given DMA address.
2. Use `LD_PRELOAD` to intercept the `close()` call and preserve the fd.

Option 1 is cleaner and should be included in the kernel patch preemptively — it's ~15 lines of code and eliminates the fragility of proc scanning entirely. Even if the library keeps the fd open today, a future NXP update could change that behavior.

---

## Delegate Implementation

### fd Discovery

After `neutronDataSetup()` returns, the Neutron Driver has created buffer fds internally and mmap'd them. The delegate needs to discover these fds. Two approaches:

**Approach A — `/proc/self/fd` scanning**: After the kernel patch, the buffer fds will be dma_buf fds. The delegate can scan `/proc/self/fd` for fds whose `readlink` target contains `dmabuf` and whose backing exporter is `neutron`. Match the mmap'd virtual address (from `dcfg.inputs[]`/`dcfg.outputs[]`) against the mmap regions in `/proc/self/maps` to associate fds with tensors.

**Approach B — Intercept the ioctl**: Use `LD_PRELOAD` or a thin wrapper to intercept `NEUTRON_IOCTL_BUFFER_CREATE` calls and capture the returned fd before `libNeutronDriver.so` processes it. This is more invasive but more reliable.

**Approach C — Direct ioctl replay**: The delegate knows the buffer sizes from TFLite tensor metadata. After `neutronDataSetup()`, it can directly call `NEUTRON_IOCTL_BUFFER_CREATE` to inspect the fd mapping. However, this creates additional buffers rather than discovering existing ones.

**Recommended: Approach A** for Phase 1. It's non-invasive, requires no interposition, and works with the unmodified proprietary library. The `/proc/self/maps` → `/proc/self/fd` correlation is straightforward: find the mmap region containing the `dcfg.inputs[0]` virtual address, extract the fd from the mapping entry, verify it's a dma_buf via `ioctl(fd, DMA_BUF_IOCTL_SYNC, ...)` succeeding.

### New Source File: `neutron_delegate_dmabuf.cc`

Responsibilities:
- After `Prepare()` completes with zero-copy enabled, scan for buffer fds
- Build a tensor_index → {fd, offset, size} mapping
- Export the `hal_dmabuf_*` C symbols from the delegate `.so`

### Integration with Existing Zero-Copy

The dmabuf feature builds on top of the existing zero-copy path (`NEUTRON_ENABLE_ZERO_COPY=1`):

1. `neutronModelPrepare()` — registers model (unchanged)
2. `neutronDataSetup()` — allocates buffers, fills `dcfg` (unchanged)
3. `SetCustomAllocationForTensor()` — registers with TFLite (unchanged)
4. **NEW**: Scan `/proc/self/fd` + `/proc/self/maps` to discover buffer fds
5. **NEW**: Build tensor → {fd, offset, size} mapping
6. **NEW**: Export `hal_dmabuf_*` symbols

The existing `Eval()` path is completely unchanged. The dmabuf API is read-only metadata — it tells the application where the buffers are, it doesn't change how inference runs.

---

## HAL DMA-BUF API

### Overview

The `hal_dmabuf_*` API is a delegate-agnostic interface exported from any TFLite delegate `.so` that supports DMA-BUF buffer sharing. The `edgefirst-tflite` Rust library discovers it via `dlsym` at runtime. Both the Neutron delegate and VX delegate export the same symbols.

The interface is formalized in the EdgeFirst HAL (`github.com/EdgeFirst/hal`) and follows existing HAL naming conventions: `hal_dmabuf_*` for functions, `HAL_DMABUF_*` for constants.

### C Header

```c
#ifndef HAL_DMABUF_H
#define HAL_DMABUF_H

#include <stdbool.h>
#include <stddef.h>

/* Opaque TFLite delegate pointer */
struct TfLiteDelegate;

/**
 * struct hal_dmabuf_tensor_info - DMA-BUF backing info for a tensor
 * @fd:     DMA-BUF file descriptor (-1 if not backed by dmabuf).
 *          Owned by the delegate — caller must NOT close it.
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
 * This is the primary probe function — if dlsym finds this symbol AND
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
 *   EINVAL  — delegate is NULL, info is NULL, info_size is 0, or
 *             tensor_index is out of range
 *   ENOENT  — tensor exists but is not backed by a DMA-BUF
 *   ENOTSUP — dmabuf not supported or not yet initialized
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
 * IMPORTANT: For GPU-to-NPU pipelines, glFinish() ensures GPU writes are
 * complete but does NOT handle CPU cache coherency. If the CPU mmap is
 * cacheable (default), you must also call hal_dmabuf_sync_for_device() after
 * glFinish() and before neutronRunBlocking(). If the buffer is mapped
 * non-cacheable (e.g., DMA_ATTR_WRITE_COMBINE), glFinish() alone suffices.
 *
 * Uses NEUTRON_IOCTL_CACHE_SYNC with precise offset/size to avoid flushing
 * the entire DMA buffer. Falls back to DMA_BUF_IOCTL_SYNC if unavailable.
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
 * Uses NEUTRON_IOCTL_CACHE_SYNC with precise offset/size to avoid
 * invalidating the entire DMA buffer.
 *
 * Returns 0 on success, -1 on error (sets errno).
 */
int hal_dmabuf_sync_for_cpu(struct TfLiteDelegate *delegate,
                            int tensor_index);

#endif /* HAL_DMABUF_H */
```

### API Design Rationale

- **Struct-based `get_tensor_info`** — Single call returns fd, offset, and size. Eliminates the ambiguous "returns 0" problem (offset=0 could mean either "first tensor" or "no dmabuf"). The caller checks the return code, not individual field values. The `info_size` parameter provides ABI stability — if the struct grows in future versions, old callers pass a smaller size and only get the fields they know about.
- **Returns 0/-1 with errno** — Consistent with the EdgeFirst HAL's existing C API conventions and POSIX patterns. Clear error discrimination via errno.
- **Per-tensor sync uses NEUTRON_IOCTL_CACHE_SYNC** — The kernel's `NEUTRON_IOCTL_CACHE_SYNC` accepts `{fd, offset, size}` for precise cache maintenance. This avoids the performance cost of `DMA_BUF_IOCTL_SYNC` which flushes the entire buffer (potentially tens of MB of weights/microcode when you only need to sync a 1.2 MB input tensor).

### Usage Example: OpenGL Preprocessing → NPU Inference

```c
/* After interpreter setup and AllocateTensors() */
TfLiteDelegate *delegate = hal_dmabuf_get_instance();

if (!hal_dmabuf_is_supported(delegate)) {
    /* Fall back to memcpy-based preprocessing */
}

/* Get NPU input buffer as dmabuf */
int input_idx = interpreter->inputs()[0];
struct hal_dmabuf_tensor_info input_info;
if (hal_dmabuf_get_tensor_info(delegate, input_idx, &input_info,
                                sizeof(input_info)) < 0) {
    /* Fall back to memcpy — tensor not backed by dmabuf */
}

/* Create EGLImage from NPU input buffer */
EGLint attrs[] = {
    EGL_WIDTH, input_width,
    EGL_HEIGHT, input_height,
    EGL_LINUX_DRM_FOURCC_EXT, DRM_FORMAT_R8,  /* or appropriate format */
    EGL_DMA_BUF_PLANE0_FD_EXT, input_info.fd,
    EGL_DMA_BUF_PLANE0_OFFSET_EXT, input_info.offset,
    EGL_DMA_BUF_PLANE0_PITCH_EXT, input_width * channels,
    EGL_NONE
};
EGLImage npu_image = eglCreateImageKHR(display, EGL_NO_CONTEXT,
                                        EGL_LINUX_DMA_BUF_EXT, NULL, attrs);

/* Bind as GL renderbuffer target */
glBindRenderbuffer(GL_RENDERBUFFER, npu_rbo);
glEGLImageTargetRenderbufferStorageOES(GL_RENDERBUFFER, npu_image);
glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                          GL_RENDERBUFFER, npu_rbo);

/* Render: sample from camera texture, write to NPU buffer */
glUseProgram(preprocess_shader);
glBindTexture(GL_TEXTURE_EXTERNAL_OES, camera_texture);
glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
glFinish();  /* Ensure GPU writes are complete */

/* Sync: flush any stale CPU cache lines before NPU reads.
 * Required when CPU mmap is cacheable (default). If using non-cacheable
 * mappings (DMA_ATTR_WRITE_COMBINE), this call is a no-op. */
hal_dmabuf_sync_for_device(delegate, input_idx);

/* Run inference — NPU reads directly from the buffer GPU just wrote to */
interpreter->Invoke();

/* Access output via dmabuf (e.g., pass to display or next model) */
int output_idx = interpreter->outputs()[0];
struct hal_dmabuf_tensor_info output_info;
hal_dmabuf_get_tensor_info(delegate, output_idx, &output_info,
                           sizeof(output_info));
/* output_info.fd and output_info.offset are ready for downstream use */
```

---

## Compatibility

### Mali GPU Compatibility (Confirmed)

| Property | Value |
|----------|-------|
| GPU | ARM Mali Valhall (`4d900000.gpu`) |
| Driver | `mali` (ARM proprietary, `libmali.so 0.54.1`) |
| DRM render node | `/dev/dri/renderD128` |
| EGL dmabuf import | `EGL_EXT_image_dma_buf_import` ✓ |
| EGL dmabuf modifiers | `EGL_EXT_image_dma_buf_import_modifiers` ✓ |
| IOMMU | Mali has its own MMU (not behind SoC SMMU) |
| CMA compatibility | Mali can map any CMA pages via `dma_buf_attach` + `dma_buf_map_attachment` |

### Neutron NPU Buffer Properties

| Property | Value |
|----------|-------|
| Allocation | `dma_alloc_attrs(dev, size, GFP_KERNEL, DMA_ATTR_FORCE_CONTIGUOUS)` |
| Memory type | CMA (960 MB pool on imx95-evk) |
| IOMMU | SMMU at `490d0000.iommu` (iommu group 9) |
| Buffer topology | Single contiguous buffer, all tensors at offsets |
| Current fd type | `anon_inode_getfd("neutron-buffer")` — NOT a dmabuf |
| After patch | `dma_buf_export()` — proper dmabuf, importable by Mali |

### Backward Compatibility

The kernel patch changes the fd type but preserves all existing behavior:
- `libNeutronDriver.so` mmaps the fd — dma_buf fds support mmap identically
- `NEUTRON_IOCTL_CACHE_SYNC` continues to work via the kernel driver's ioctl handler
- `neutron_buffer_get_from_fd()` needs to use `dma_buf_get(fd)->priv` instead of `file->private_data`
- Existing applications that don't use `hal_dmabuf_*` see no change

---

## Open Questions and Risks

### P0 — Must resolve before implementation

1. **Does `libNeutronDriver.so` close the buffer fd after mmap?** If the library calls `close(fd)` after `mmap()`, the fd disappears from `/proc/self/fd` and Approach A (fd scanning) is dead. The mmap itself survives (it holds a kernel reference), but the fd is gone. **Must verify on target** by inspecting `/proc/<pid>/fd` during inference with the unmodified delegate. If confirmed closed, fall back to Approach B (ioctl interception) or add `NEUTRON_IOCTL_GET_BUFFER_FD` to the kernel patch.

### P1 — Resolve during implementation

2. **fd discovery algorithm**: The single-buffer-per-NeutronGraph topology simplifies discovery. For the common case, there is exactly one neutron dmabuf fd per delegated partition. The algorithm: scan `/proc/self/fd`, identify neutron dmabufs by readlink target (`/dmabuf:neutron` on kernel 6.12+, requires `exp_info.exp_name = "neutron"`), correlate with `/proc/self/maps` by inode number to find the mmap base address. Tensor offset = `dcfg.inputs[i] - mmap_base`. Phase 2 improvement: add `NEUTRON_IOCTL_GET_BUFFER_FD` for a clean non-fragile path.

3. **Tensor offset calculation**: The delegate knows tensor virtual addresses (from `dcfg.inputs[]`/`dcfg.outputs[]`) and the buffer's base virtual address (from the mmap region start). The offset is `tensor_vaddr - mmap_base`. This assumes all activation tensors are within one mmap — validated by the single-buffer architecture.

4. **Multi-NeutronGraph models**: Models with multiple `NeutronGraph` nodes may have separate buffers per node. Each `NeutronDelegateKernel` operation has its own `dcfg`, so fd discovery should be scoped per-operation after each `neutronDataSetup()` call.

5. **Cache coherency in GPU→NPU path**: The Neutron kernel driver sets `ndev->dev->dma_coherent = true` at init but toggles it to `false` around every `neutron_memory_sync()` call — the NPU is NOT hardware-coherent. The CPU mmap is cacheable by default (`dma_mmap_attrs` without `DMA_ATTR_WRITE_COMBINE`). For the GPU→NPU path, this means stale CPU cache lines could corrupt GPU-written data. Two mitigations: (a) use `DMA_BUF_IOCTL_SYNC` between GPU write and NPU read, or (b) request `DMA_ATTR_WRITE_COMBINE` for activation tensor mappings to eliminate CPU caching entirely (preferred since we don't want CPU access).

6. **Whole-buffer vs. per-tensor cache sync**: `DMA_BUF_IOCTL_SYNC` operates on the entire dmabuf (potentially tens of MB). The `NEUTRON_IOCTL_CACHE_SYNC` accepts offset + size for precise scoping. The `hal_dmabuf_sync_*` implementation should prefer the latter to avoid flushing the entire buffer when only a 1.2 MB input tensor needs syncing.

### P2 — Resolve during integration

7. **EGLImage format mapping**: The NPU input tensor is typically quantized INT8 in NHWC layout. The EGLImage creation needs the correct `DRM_FORMAT_*` fourcc and pitch. For 3-channel INT8 (RGB), `DRM_FORMAT_BGR888` or `DRM_FORMAT_RGB888` applies. Must validate on target that Mali accepts the chosen fourcc for renderbuffer usage and that the data layout matches the NPU's expectations.

8. **EGLImage non-zero offset support**: Since tensors are packed in a single buffer, input tensor EGLImages use `EGL_DMA_BUF_PLANE0_OFFSET_EXT != 0`. The `EGL_EXT_image_dma_buf_import` spec mandates offset support, but driver bugs exist in the wild. Must validate on target with Mali Valhall.

9. **Kernel `devm_kzalloc` lifetime risk**: The `neutron_buffer` struct is allocated with `devm_kzalloc` (device-managed). With dma_buf export, another device (Mali) may hold a `dma_buf_attach` reference that outlives the neutron device. If the neutron device is unbound, the struct is freed but the dma_buf still references it — use-after-free. Future hardening: use `kzalloc` and manage lifetime through kref/dma_buf release.

---

## References

- **VX Delegate DMABUF.md**: `tflite-vx-delegate-imx/DMABUF.md` — prior art for dmabuf support in TFLite delegate
- **Neutron kernel driver**: `linux-imx/drivers/staging/neutron/` — GPL source, target of kernel patch
- **Neutron userspace driver**: `NeutronDriver.h` — public API (BSD-3-Clause header, proprietary binary)
- **EdgeFirst HAL**: `github.com/EdgeFirst/hal` — HAL tensor/dmabuf abstractions
- **EdgeFirst TFLite**: `github.com/EdgeFirst/tflite-rs` — Rust TFLite bindings with delegate probing
- **meta-edgefirst**: `github.com/EdgeFirstAI/meta-edgefirst` — Yocto layer for kernel patches
- **EGL dmabuf import**: `EGL_EXT_image_dma_buf_import` extension specification
