// Holds a given amount of touched anonymous memory until a sentinel file is removed, to put the
// machine in a known memory state for the host planner. It grows in steps and stops growing when
// the system nears trouble: critical pressure, swap growing, or little free memory left.
// usage: memhold GIB SENTINEL [--mlock] [--min-free-gib N] [--max-swap-growth-gib N]
#include <mach/mach.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <unistd.h>

static const double GiB = 1073741824.0;

static double free_gib(void) {
    vm_statistics64_data_t vs;
    mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vs, &n) != KERN_SUCCESS) return -1;
    vm_size_t page = 0;
    host_page_size(mach_host_self(), &page);
    return (double) (vs.free_count + vs.speculative_count + vs.purgeable_count) * page / GiB;
}

static double swap_gib(void) {
    struct xsw_usage sw;
    size_t len = sizeof(sw);
    return sysctlbyname("vm.swapusage", &sw, &len, NULL, 0) == 0 ? sw.xsu_used / GiB : -1;
}

static int pressure(void) {
    int lvl = 0;
    size_t len = sizeof(lvl);
    return sysctlbyname("kern.memorystatus_vm_pressure_level", &lvl, &len, NULL, 0) == 0 ? lvl : -1;
}

static double rss_gib(void) {
    struct mach_task_basic_info info;
    mach_msg_type_number_t n = MACH_TASK_BASIC_INFO_COUNT;
    return task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t) &info, &n) == KERN_SUCCESS ? info.resident_size / GiB : -1;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: memhold GIB SENTINEL [--mlock] [--min-free-gib N] [--max-swap-growth-gib N]\n"); return 1; }
    const size_t want = (size_t) (atof(argv[1])*GiB);
    int lock = 0;
    double min_free = 1.5, max_swap = 0.5;
    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--mlock") == 0) lock = 1;
        else if (strcmp(argv[i], "--min-free-gib") == 0 && i + 1 < argc) min_free = atof(argv[++i]);
        else if (strcmp(argv[i], "--max-swap-growth-gib") == 0 && i + 1 < argc) max_swap = atof(argv[++i]);
    }
    char * p = want ? mmap(NULL, want, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0) : NULL;
    if (want && p == MAP_FAILED) { perror("mmap"); return 1; }
    const double swap0 = swap_gib();
    const size_t step = (size_t) 256 << 20;
    size_t held = 0;
    const char * stop = "target";
    while (held < want) {
        const size_t n = want - held < step ? want - held : step;
        for (size_t i = 0; i < n; i += 4096) p[held + i] = (char) ((held + i) >> 12);
        if (lock && mlock(p + held, n) != 0) { stop = "mlock failed"; break; }
        held += n;
        const int lvl = pressure();
        if (lvl >= 4) { stop = "critical pressure"; break; }
        if (swap_gib() - swap0 > max_swap) { stop = "swap growing"; break; }
        if (free_gib() >= 0 && free_gib() < min_free) { stop = "free memory low"; break; }
    }
    FILE * f = fopen(argv[2], "w");
    if (f) fclose(f);
    printf("holding %.2f of %.2f GiB (%s)%s | rss %.2f GiB, pressure %d, swap %+.2f GiB\n", held/GiB, want/GiB, stop,
           lock ? ", wired" : "", rss_gib(), pressure(), swap_gib() - swap0);
    fflush(stdout);
    while (access(argv[2], F_OK) == 0) {
        // keep the pages active so the system treats them as in use
        for (size_t i = 0; i < held; i += 4096) p[i]++;
        sleep(2);
    }
    if (held) munmap(p, want);
    printf("released\n");
    return 0;
}
