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

#include "neutron_delegate_dmabuf.h"
#include "hal_dmabuf.h"

#include <dirent.h>
#include <cerrno>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cinttypes>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

/* linux/dma-buf.h may not be available in all cross-compile sysroots */
#ifndef DMA_BUF_SYNC_READ
struct dma_buf_sync {
    uint64_t flags;
};
#define DMA_BUF_SYNC_READ      (1 << 0)
#define DMA_BUF_SYNC_WRITE     (2 << 0)
#define DMA_BUF_SYNC_RW        (DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE)
#define DMA_BUF_SYNC_START     (0 << 2)
#define DMA_BUF_SYNC_END       (1 << 2)
#define DMA_BUF_BASE            'b'
#define DMA_BUF_IOCTL_SYNC     _IOW(DMA_BUF_BASE, 0, struct dma_buf_sync)
#endif

using namespace std;

/* ── Internal state ─────────────────────────────────────────────────── */

struct DmabufTensorEntry {
    int fd;
    size_t offset;
    size_t size;
};

static struct DmabufGlobal {
    TfLiteDelegate *delegate = nullptr;
    bool discovered = false;
    unordered_map<int, DmabufTensorEntry> tensors;
} g_dmabuf;

/* ── Helpers for /proc parsing ──────────────────────────────────────── */

struct NeutronFdInfo {
    int fd;
    ino_t inode;
    size_t buf_size;
};

/*
 * Scan /proc/self/fd for DMA-BUF fds exported by the neutron driver.
 * A candidate fd has readlink target containing "/dmabuf:" and
 * /proc/self/fdinfo/<fd> containing "exp_name:\tneutron".
 */
static vector<NeutronFdInfo> find_neutron_fds()
{
    vector<NeutronFdInfo> result;

    DIR *dir = opendir("/proc/self/fd");
    if (!dir)
        return result;
    int dir_fd = dirfd(dir);

    struct dirent *ent;
    while ((ent = readdir(dir)) != nullptr) {
        /* Skip non-numeric entries */
        char *endp;
        int fd = strtol(ent->d_name, &endp, 10);
        if (*endp != '\0')
            continue;

        /* Skip the fd for the directory handle itself */
        if (fd == dir_fd)
            continue;

        /* readlink to check if this is a dmabuf */
        char link_path[64];
        char link_target[256];
        snprintf(link_path, sizeof(link_path), "/proc/self/fd/%d", fd);
        ssize_t len = readlink(link_path, link_target, sizeof(link_target) - 1);
        if (len <= 0)
            continue;
        link_target[len] = '\0';

        if (strstr(link_target, "/dmabuf:") == nullptr)
            continue;

        /* Read fdinfo to confirm exp_name is "neutron" and get size */
        char fdinfo_path[64];
        snprintf(fdinfo_path, sizeof(fdinfo_path), "/proc/self/fdinfo/%d", fd);
        ifstream fdinfo(fdinfo_path);
        if (!fdinfo.is_open())
            continue;

        bool is_neutron = false;
        size_t buf_size = 0;
        string line;
        while (getline(fdinfo, line)) {
            /* Exact match: "exp_name:\tneutron" */
            if (line == "exp_name:\tneutron") {
                is_neutron = true;
            }
            if (line.compare(0, 5, "size:") == 0) {
                buf_size = strtoull(line.c_str() + 5, nullptr, 10);
            }
        }

        if (!is_neutron)
            continue;

        /* Get inode for /proc/self/maps correlation */
        struct stat st;
        if (fstat(fd, &st) != 0)
            continue;

        result.push_back({fd, st.st_ino, buf_size});
    }

    closedir(dir);
    return result;
}

/*
 * Parse /proc/self/maps and return regions whose inode matches one of
 * the neutron dmabuf fds.
 */
static unordered_map<int, uintptr_t> find_mmap_bases(
    const vector<NeutronFdInfo> &fds)
{
    /* Build inode → fd lookup */
    unordered_map<ino_t, int> inode_to_fd;
    for (auto &info : fds)
        inode_to_fd[info.inode] = info.fd;

    unordered_map<int, uintptr_t> fd_to_base;

    ifstream maps("/proc/self/maps");
    if (!maps.is_open())
        return fd_to_base;

    string line;
    while (getline(maps, line)) {
        /*
         * Format: start-end perms offset major:minor inode [pathname]
         * Example: 7f8000000-7f8a00000 rw-s 00000000 00:07 12345 /dmabuf:neutron
         */
        uintptr_t start, end;
        unsigned long long offset_val;
        unsigned int major_val, minor_val;
        unsigned long inode_val;

        int parsed = sscanf(line.c_str(),
                            "%" SCNxPTR "-%" SCNxPTR " %*s %llx %x:%x %lu",
                            &start, &end, &offset_val, &major_val,
                            &minor_val, &inode_val);
        if (parsed < 6)
            continue;

        ino_t inode = (ino_t)inode_val;
        auto it = inode_to_fd.find(inode);
        if (it == inode_to_fd.end())
            continue;

        /* Only record the first (lowest-address) mapping per fd */
        fd_to_base.emplace(it->second, start);
    }

    return fd_to_base;
}

/* ── Internal API called from neutron_delegate.cc ───────────────────── */

void dmabuf_set_delegate(TfLiteDelegate *delegate)
{
    if (g_dmabuf.delegate != nullptr && g_dmabuf.delegate != delegate) {
        cerr << "WARNING: dmabuf_set_delegate called with existing delegate — "
                "previous dmabuf state cleared" << endl;
    }
    g_dmabuf.delegate = delegate;
    g_dmabuf.discovered = false;
    g_dmabuf.tensors.clear();
}

void dmabuf_clear()
{
    g_dmabuf.delegate = nullptr;
    g_dmabuf.discovered = false;
    g_dmabuf.tensors.clear();
}

void dmabuf_discover(const vector<DmabufTensorVaddr> &tensor_vaddrs)
{
    /* Phase 1: Find neutron dmabuf fds */
    vector<NeutronFdInfo> neutron_fds = find_neutron_fds();
    if (neutron_fds.empty()) {
        /* No neutron dmabufs found — running without kernel patch or
         * fds were closed. Silently degrade. */
        return;
    }

    /* Phase 2: Correlate fds with mmap regions */
    unordered_map<int, uintptr_t> fd_to_base = find_mmap_bases(neutron_fds);
    if (fd_to_base.empty()) {
        cerr << "WARNING: neutron dmabuf fds found but no matching mmap "
                "regions in /proc/self/maps" << endl;
        return;
    }

    /* Build a sorted list of regions for efficient lookup */
    struct FdRegion {
        int fd;
        uintptr_t base;
        size_t size;
    };
    vector<FdRegion> regions;
    for (auto &info : neutron_fds) {
        auto it = fd_to_base.find(info.fd);
        if (it != fd_to_base.end())
            regions.push_back({info.fd, it->second, info.buf_size});
    }

    /* Phase 3: Map each tensor vaddr to its containing fd region */
    int mapped_count = 0;
    for (auto &tv : tensor_vaddrs) {
        /* First-writer-wins: skip if already mapped (shared tensors) */
        if (g_dmabuf.tensors.find(tv.tensor_index) != g_dmabuf.tensors.end())
            continue;

        for (auto &region : regions) {
            if (tv.vaddr >= region.base &&
                tv.vaddr < region.base + region.size) {
                size_t offset = tv.vaddr - region.base;
                g_dmabuf.tensors[tv.tensor_index] = {
                    region.fd, offset, tv.size};
                mapped_count++;
                break;
            }
        }
    }

    if (mapped_count > 0) {
        g_dmabuf.discovered = true;
        cout << "INFO: dmabuf discovery mapped " << mapped_count
             << " tensors across " << regions.size() << " buffer(s)" << endl;
    }
}

/* ── DMA-BUF sync helper ───────────────────────────────────────────── */

static int dmabuf_sync(int fd, uint64_t flags)
{
    struct dma_buf_sync sync;
    sync.flags = flags;
    return ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
}

/* ── hal_dmabuf_* public API ────────────────────────────────────────── */

static const DmabufTensorEntry *dmabuf_lookup(TfLiteDelegate *delegate,
                                               int tensor_index)
{
    if (!delegate) {
        errno = EINVAL;
        return nullptr;
    }
    if (g_dmabuf.delegate != delegate || !g_dmabuf.discovered) {
        errno = ENOTSUP;
        return nullptr;
    }
    auto it = g_dmabuf.tensors.find(tensor_index);
    if (it == g_dmabuf.tensors.end()) {
        errno = ERANGE;
        return nullptr;
    }
    return &it->second;
}

extern "C" {

__attribute__((visibility("default")))
hal_delegate_t hal_dmabuf_get_instance(void)
{
    return static_cast<hal_delegate_t>(g_dmabuf.delegate);
}

__attribute__((visibility("default")))
int hal_dmabuf_is_supported(hal_delegate_t delegate)
{
    TfLiteDelegate *d = static_cast<TfLiteDelegate *>(delegate);
    return (g_dmabuf.delegate != nullptr &&
            g_dmabuf.delegate == d &&
            g_dmabuf.discovered) ? 1 : 0;
}

__attribute__((visibility("default")))
int hal_dmabuf_get_tensor_info(hal_delegate_t delegate,
                               int tensor_index,
                               hal_dmabuf_tensor_info *info,
                               size_t info_size)
{
    if (!info || info_size < sizeof(hal_dmabuf_tensor_info) || tensor_index < 0) {
        errno = EINVAL;
        return -1;
    }
    TfLiteDelegate *d = static_cast<TfLiteDelegate *>(delegate);
    const DmabufTensorEntry *entry = dmabuf_lookup(d, tensor_index);
    if (!entry)
        return -1;

    memset(info, 0, info_size);
    info->fd = entry->fd;
    info->offset = entry->offset;
    info->size = entry->size;
    /* Shape/dtype not available from DmabufTensorEntry; set ndim = 0 per spec */
    info->ndim = 0;
    info->dtype = HAL_DTYPE_U8;
    return 0;
}

__attribute__((visibility("default")))
int hal_dmabuf_sync_for_device(hal_delegate_t delegate,
                               int tensor_index)
{
    if (tensor_index < 0) {
        errno = EINVAL;
        return -1;
    }
    TfLiteDelegate *d = static_cast<TfLiteDelegate *>(delegate);
    const DmabufTensorEntry *entry = dmabuf_lookup(d, tensor_index);
    if (!entry)
        return -1;

    /* Flush CPU caches after CPU writes, before device reads */
    return dmabuf_sync(entry->fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);
}

__attribute__((visibility("default")))
int hal_dmabuf_sync_for_cpu(hal_delegate_t delegate,
                            int tensor_index)
{
    if (tensor_index < 0) {
        errno = EINVAL;
        return -1;
    }
    TfLiteDelegate *d = static_cast<TfLiteDelegate *>(delegate);
    const DmabufTensorEntry *entry = dmabuf_lookup(d, tensor_index);
    if (!entry)
        return -1;

    /* Invalidate CPU caches before reading device-written data */
    return dmabuf_sync(entry->fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);
}

__attribute__((visibility("default")))
int hal_camera_adaptor_is_supported(hal_delegate_t delegate,
                                    const char *format)
{
    (void)delegate;
    (void)format;
    return 0;
}

__attribute__((visibility("default")))
int hal_camera_adaptor_get_format_info(hal_delegate_t delegate,
                                       const char *format,
                                       hal_camera_adaptor_format_info *info,
                                       size_t info_size)
{
    (void)delegate;
    if (!format || !info || info_size == 0) {
        errno = EINVAL;
        return -1;
    }
    errno = ENOTSUP;
    return -1;
}

}  /* extern "C" */
