// test_memop_isolated.cpp — 单操作隔离测试：memset 或 memcpy，可配置大小和模式
//
// 目的：以最少干扰测量单一操作的吞吐，支持两种模式：
//   1. 默认模式（pre-faulted）：预先 touch 所有页面，测纯内存操作速度
//   2. first-touch 模式：每次迭代 mmap 全新页面，测包含 page fault 的真实耗时
//
// first-touch 模式用 mmap/munmap 而非 malloc/free，因为 mmap 的
// MAP_ANONYMOUS 保证每次都是全新的虚拟页（无物理页映射），而 malloc 可能
// 复用 glibc 缓存的内存块，page fault 行为不可控。
//
// 用法：./test_memop_isolated <memset|memcpy> [size_mb] [iters] [--first-touch]
// 编译：g++ -O3 -o test_memop_isolated test_memop_isolated.cpp

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <sys/mman.h>

// 编译器屏障，零运行时开销。原理详见 test_final_fair.cpp。
// volatile: 不能删除/重排; "g"(p): p 及其上游计算不能被消除;
// "memory": 所有内存写入必须落地，阻止编译器缓存优化。
void doNotOptimize(void* p) {
    asm volatile("" : : "g"(p) : "memory");
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <memset|memcpy> [size_mb] [iters] [--first-touch]\n", argv[0]);
        return 1;
    }

    size_t size_mb = argc >= 3 ? atoi(argv[2]) : 64;
    int iters = argc >= 4 ? atoi(argv[3]) : 100;
    size_t sz = size_mb * 1024ULL * 1024;
    int first_touch = (argc >= 5 && strcmp(argv[4], "--first-touch") == 0);

    if (first_touch) {
        // First-touch 模式：每次迭代 mmap 全新匿名页，memset 触发真实 page fault。
        // munmap 后内核完全回收，保证每次迭代的 PF 行为一致。
        double total_ms = 0;
        for (int i = 0; i < iters; i++) {
            char* buf = (char*)mmap(NULL, sz, PROT_READ|PROT_WRITE,
                                    MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
            auto t0 = std::chrono::high_resolution_clock::now();
            memset(buf, 0, sz);
            doNotOptimize(buf);
            auto t1 = std::chrono::high_resolution_clock::now();
            total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            munmap(buf, sz);
        }
        printf("first-touch memset %zuMB x %d: %.3f ms/iter\n",
               size_mb, iters, total_ms / iters);
    } else {
        // Pre-faulted 模式：预先 touch 所有页面，后续循环测纯内存操作速度。
        // dst 填非零值确保物理页已映射且不是 CoW 零页。
        char* dst = (char*)malloc(sz);
        memset(dst, 1, sz);
        doNotOptimize(dst);

        char* src = (char*)malloc(sz);
        memset(src, 0, sz);
        doNotOptimize(src);

        auto t0 = std::chrono::high_resolution_clock::now();
        if (strcmp(argv[1], "memset") == 0) {
            for (int i = 0; i < iters; i++) {
                memset(dst, 0, sz);
                doNotOptimize(dst);
            }
        } else {
            for (int i = 0; i < iters; i++) {
                memcpy(dst, src, sz);
                doNotOptimize(dst);
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("%s %zuMB x %d: %.3f ms total, %.3f ms/iter\n",
               argv[1], size_mb, iters, ms, ms / iters);

        free(dst);
        free(src);
    }
    return 0;
}
