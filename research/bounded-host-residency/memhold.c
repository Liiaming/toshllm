// Holds a given amount of touched anonymous memory until a sentinel file is removed, to put the
// machine in a known memory state for the host planner.
// usage: memhold GIB SENTINEL
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: memhold GIB SENTINEL\n"); return 1; }
    const size_t bytes = (size_t) (atof(argv[1])*1073741824.0);
    char * p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); return 1; }
    for (size_t i = 0; i < bytes; i += 4096) p[i] = (char) (i >> 12);
    FILE * f = fopen(argv[2], "w");
    if (f) fclose(f);
    printf("holding %.2f GiB\n", bytes/1073741824.0);
    fflush(stdout);
    while (access(argv[2], F_OK) == 0) {
        // keep the pages active so the system treats them as in use
        for (size_t i = 0; i < bytes; i += 4096) p[i]++;
        sleep(2);
    }
    munmap(p, bytes);
    printf("released\n");
    return 0;
}
