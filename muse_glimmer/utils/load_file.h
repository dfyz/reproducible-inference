#pragma once

#include <err.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

void* load_file(const char* file_name, size_t n_bytes) {
    // We rely on the OS unmapping the file and closing the FD.
    int fd = open(file_name, O_RDONLY);
    if (fd < 0) {
        err(1, "Failed to open %s", file_name);
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        err(2, "Failed to fstat");
    }

    if (st.st_size != n_bytes) {
        errx(3, "Expected a file of %zu bytes, got %zu\n", n_bytes, st.st_size);
    }

    void* res = mmap(NULL, n_bytes, PROT_READ, MAP_SHARED, fd, 0);
    if (res == MAP_FAILED) {
        err(4, "Failed to mmap");
    }
    return res;
}
