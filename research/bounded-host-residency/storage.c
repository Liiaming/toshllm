// Read latency of expert-sized ranges from a model file.
// usage: storage FILE MODE SIZE_KIB COUNT [SEED]
//   MODE nocache : pread with F_NOCACHE, the storage device itself
//        cache   : pread of ranges read once just before, the page cache
//        mmap    : first touch of mmap'd pages the process never read (page faults)
//        seq     : sequential pread with F_NOCACHE from a random start
// Prints one line: mode size count p50 p95 p99 (ms) and GB/s.
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec*1e3 + t.tv_nsec/1e6;
}

static int cmp(const void * a, const void * b) {
    const double x = *(const double *) a, y = *(const double *) b;
    return x < y ? -1 : x > y;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: storage FILE MODE SIZE_KIB COUNT [SEED]\n");
        return 1;
    }
    const char * mode = argv[2];
    const size_t size = (size_t) atol(argv[3])*1024;
    const int n = atoi(argv[4]);
    srand(argc > 5 ? atoi(argv[5]) : 1);

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    fstat(fd, &st);
    const size_t page = 16384;
    char * buf = aligned_alloc(page, size + page);
    double * lat = calloc(n, sizeof(double));

    if (strcmp(mode, "nocache") == 0 || strcmp(mode, "seq") == 0) {
        fcntl(fd, F_NOCACHE, 1);
    }
    char * map = NULL;
    if (strcmp(mode, "mmap") == 0) {
        map = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (map == MAP_FAILED) { perror("mmap"); return 1; }
    }

    off_t seq_off = ((off_t) ((double) rand()/RAND_MAX*(st.st_size - (off_t) size*n - 1)))/page*page;
    double total = 0;
    volatile unsigned long sink = 0;
    for (int i = 0; i < n; i++) {
        off_t off = ((off_t) ((double) rand()/RAND_MAX*(st.st_size - size - 1)))/page*page;
        if (strcmp(mode, "seq") == 0) off = seq_off + (off_t) i*size;
        if (strcmp(mode, "cache") == 0) {
            pread(fd, buf, size, off);   // brings the range into the page cache
        }
        const double t0 = now_ms();
        if (map) {
            for (size_t p = 0; p < size; p += 4096) sink += (unsigned char) map[off + p];
        } else {
            ssize_t r = pread(fd, buf, size, off);
            if (r != (ssize_t) size) { perror("pread"); return 1; }
        }
        lat[i] = now_ms() - t0;
        total += lat[i];
    }
    qsort(lat, n, sizeof(double), cmp);
    printf("%s %zu KiB n=%d p50 %.3f p95 %.3f p99 %.3f ms | %.2f GB/s\n", mode, size/1024, n,
           lat[n/2], lat[(int) (n*0.95)], lat[(int) (n*0.99)], (double) size*n/(total/1e3)/1e9);
    (void) sink;
    return 0;
}
