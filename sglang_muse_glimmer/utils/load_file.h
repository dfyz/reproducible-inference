#pragma once

#include <err.h>
#include <fcntl.h>
#include <sys/mman.h>

void* load_file(const char* file_name, size_t n_bytes) {
    // We rely on the OS unmapping the file and closing the FD.
    int fd = open(file_name, O_RDONLY);
    if (fd < 0) {
        err(1, "Failed to open %s", file_name);
    }

    void* res = mmap(NULL, n_bytes, PROT_READ, MAP_SHARED, fd, 0);
    if (res == MAP_FAILED) {
        err(2, "Failed to mmap");
    }
    return res;
}
