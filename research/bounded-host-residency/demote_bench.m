// VRAM -> host copy of expert-sized ranges on Metal: the paths an evicted expert could take back
// to the RAM cache instead of a reread from the model file.
//   direct : blit private -> page-aligned mlocked RAM wrapped with newBufferWithBytesNoCopy
//   stage  : blit private -> small shared staging ring, then memcpy into mlocked RAM
// Each copy is its own command buffer on a queue of its own; in-flight copies 1/2/4/8. Also run
// against a busy compute queue to see whether the copies wait for compute or slow it down.
// build: clang -O2 -fobjc-arc demote_bench.m -framework Metal -framework Foundation -o demote_bench
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <stdatomic.h>
#include <sys/mman.h>

static double now_us(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double) mach_absolute_time() * tb.numer / tb.denom / 1e3;
}

static int cmp(const void * a, const void * b) { double x = *(const double *) a, y = *(const double *) b; return x < y ? -1 : x > y; }
static double pct(double * v, int n, double f) { qsort(v, n, sizeof(double), cmp); return v[(int) (f*(n - 1))]; }

static NSString * busy_src = @"#include <metal_stdlib>\nusing namespace metal;\n"
    "kernel void busy(device float * o [[buffer(0)]], uint i [[thread_position_in_grid]]) {"
    " float x = o[i]; for (int k = 0; k < 20000; ++k) x = fma(x, 1.0001f, 0.0001f); o[i] = x; }";

int main(int argc, char ** argv) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        printf("device %s, unified %d, maxBufferLength %.2f GiB, recommendedMaxWorkingSet %.2f GiB\n", dev.name.UTF8String,
               dev.hasUnifiedMemory, dev.maxBufferLength/1073741824.0, dev.recommendedMaxWorkingSetSize/1073741824.0);
        const size_t sizes[2] = { 1769472, 2039808 };     // 1.688 and 1.945 MiB experts
        const size_t arena_bytes = (size_t) 256 << 20, host_bytes = (size_t) 256 << 20, ring_bytes = (size_t) 32 << 20;
        id<MTLBuffer> arena = [dev newBufferWithLength:arena_bytes options:MTLResourceStorageModePrivate];
        char * host = mmap(NULL, host_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (mlock(host, host_bytes) != 0) { perror("mlock"); return 1; }
        id<MTLBuffer> wrapped = [dev newBufferWithBytesNoCopy:host length:host_bytes options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> ring = [dev newBufferWithLength:ring_bytes options:MTLResourceStorageModeShared];
        printf("wrap of mlocked RAM: %s; allocated after %.1f MiB\n", wrapped ? "ok" : "FAILED", dev.currentAllocatedSize/1048576.0);
        id<MTLCommandQueue> q_copy = [dev newCommandQueue], q_busy = [dev newCommandQueue];
        NSError * err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:busy_src options:nil error:&err];
        id<MTLComputePipelineState> busy = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"busy"] error:&err];
        id<MTLBuffer> busy_buf = [dev newBufferWithLength:(1 << 20)*4 options:MTLResourceStorageModePrivate];

        // how long a busy command buffer alone takes
        double busy_alone = 0;
        for (int r = 0; r < 3; ++r) {
            id<MTLCommandBuffer> cb = [q_busy commandBuffer];
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
            [ce setComputePipelineState:busy]; [ce setBuffer:busy_buf offset:0 atIndex:0];
            [ce dispatchThreads:MTLSizeMake(1 << 20, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [ce endEncoding]; [cb commit]; [cb waitUntilCompleted];
            busy_alone = (cb.GPUEndTime - cb.GPUStartTime)*1e3;
        }
        printf("busy kernel alone %.2f ms\n", busy_alone);

        const int N = 96;
        for (int path = 0; path < 2; ++path) {
          for (int loaded = 0; loaded < 2; ++loaded) {
            for (int si = 0; si < 2; ++si) {
              for (int inflight = 1; inflight <= 8; inflight *= 2) {
                const size_t sz = sizes[si];
                double enq[N], gpu[N], done[N], cpy[N];
                __block _Atomic int finished = 0;
                __block double * t_end = calloc(N, sizeof(double));
                double t_commit[N];
                id<MTLCommandBuffer> busy_cb = nil;
                if (loaded) {
                    busy_cb = [q_busy commandBuffer];
                    for (int k = 0; k < 8; ++k) {
                        id<MTLComputeCommandEncoder> ce = [busy_cb computeCommandEncoder];
                        [ce setComputePipelineState:busy]; [ce setBuffer:busy_buf offset:0 atIndex:0];
                        [ce dispatchThreads:MTLSizeMake(1 << 20, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        [ce endEncoding];
                    }
                    [busy_cb commit];
                }
                NSMutableArray * cbs = [NSMutableArray array];
                const double t0 = now_us();
                for (int i = 0; i < N; ++i) {
                    // keep at most `inflight` copies outstanding
                    while (i - atomic_load(&finished) >= inflight) { }
                    const size_t src = (size_t) (i % 64)*sz % (arena_bytes - sz);
                    const size_t dst = (size_t) (i % 64)*sz % (host_bytes - sz);
                    const size_t rs  = (size_t) (i % 8)*(ring_bytes/8);
                    const double te = now_us();
                    id<MTLCommandBuffer> cb = [q_copy commandBuffer];
                    id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
                    if (path == 0) [be copyFromBuffer:arena sourceOffset:src toBuffer:wrapped destinationOffset:dst size:sz];
                    else           [be copyFromBuffer:arena sourceOffset:src toBuffer:ring destinationOffset:rs size:sz];
                    [be endEncoding];
                    const int idx = i;
                    char * hostp = host + dst; char * ringp = (char *) ring.contents + rs;
                    const int stage = path;
                    __block double * cpyp = cpy;
                    [cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
                        const double tc = now_us();
                        if (stage) memcpy(hostp, ringp, sz);
                        cpyp[idx] = now_us() - tc;
                        t_end[idx] = now_us();
                        atomic_fetch_add(&finished, 1);
                    }];
                    t_commit[i] = now_us();
                    [cb commit];
                    enq[i] = now_us() - te;
                    [cbs addObject:cb];
                }
                while (atomic_load(&finished) < N) { }
                const double wall = now_us() - t0;
                for (int i = 0; i < N; ++i) {
                    id<MTLCommandBuffer> cb = cbs[i];
                    gpu[i] = (cb.GPUEndTime - cb.GPUStartTime)*1e6;
                    done[i] = t_end[i] - t_commit[i];
                    if (!path) cpy[i] = 0;
                }
                double busy_ms = 0;
                if (busy_cb) { [busy_cb waitUntilCompleted]; busy_ms = (busy_cb.GPUEndTime - busy_cb.GPUStartTime)*1e3; }
                printf("%-6s %-9s %.3f MiB x%d: enqueue p50 %.0f us | gpu p50 %.0f p95 %.0f us | commit->done p50 %.0f p95 %.0f p99 %.0f us | memcpy p50 %.0f us | %.2f GB/s%s",
                       path ? "stage" : "direct", loaded ? "gpu busy" : "gpu idle", sz/1048576.0, inflight, pct(enq, N, 0.5),
                       pct(gpu, N, 0.5), pct(gpu, N, 0.95), pct(done, N, 0.5), pct(done, N, 0.95), pct(done, N, 0.99), pct(cpy, N, 0.5),
                       (double) sz*N/(wall*1e3), loaded ? "" : "\n");
                if (loaded) printf(" | busy 8x %.1f ms (alone 8x %.1f)\n", busy_ms, 8*busy_alone);
                free(t_end);
              }
            }
          }
        }
        // can the whole RAM cache be exposed? wrap 12 GiB in 1 GiB pieces
        const size_t big = (size_t) 12 << 30, piece = (size_t) 1 << 30;
        char * cache = mmap(NULL, big, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        const int locked = mlock(cache, big) == 0;
        int ok = 0;
        NSMutableArray * wraps = [NSMutableArray array];
        for (size_t o = 0; o < big; o += piece) {
            id<MTLBuffer> w = [dev newBufferWithBytesNoCopy:cache + o length:piece options:MTLResourceStorageModeShared deallocator:nil];
            if (w) { ok++; [wraps addObject:w]; }
        }
        printf("12 GiB mlocked %d, wrapped %d of 12 pieces of 1 GiB, currentAllocatedSize %.2f GiB\n", locked, ok, dev.currentAllocatedSize/1073741824.0);
    }
    return 0;
}
