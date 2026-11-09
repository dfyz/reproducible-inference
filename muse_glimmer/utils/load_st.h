#pragma once

#include <err.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <stdint.h>

void* load_st(const char* file_name, void* addr) {
    // We rely on the OS unmapping the file and closing the FD.
    int fd = open(file_name, O_RDONLY);
    if (fd < 0) {
        err(1, "failed to open %s", file_name);
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        err(2, "failed to stat %s", file_name);
    }

    uint64_t h_len;
    if (read(fd, &h_len, sizeof(h_len)) != sizeof(h_len)) {
        err(3, "failed to read the header length from %s", file_name);
    }

    char* res = mmap(addr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (res == MAP_FAILED) {
        err(4, "failed to mmap %s", file_name);
    }
    return res + sizeof(h_len) + h_len;
}
