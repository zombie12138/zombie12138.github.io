// test_memop.cpp — 不同 buffer 大小下 memset vs memcpy 的性能扫描
//
// 目的：观察 memset 和 memcpy 的性能差距随 buffer 大小的变化趋势，
//       找到 L1 / L2 / L3 / 主存各级 cache 边界处的拐点。
//
// 设计：所有页面预先 touch（排除 page fault），带 warmup（稳定分支预测、
//       TLB、glibc IFUNC 分发），迭代次数按数据量自适应。
//
// 编译：g++ -O3 -o test_memop test_memop.cpp

#include <iostream>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <string>

using namespace std;

string format_size(size_t bytes) {
    if (bytes < 1024) return to_string(bytes) + " B";
    if (bytes < 1024 * 1024) return to_string(bytes / 1024) + " KB";
    if (bytes < 1024ULL * 1024 * 1024) return to_string(bytes / (1024 * 1024)) + " MB";
    return to_string(bytes / (1024ULL * 1024 * 1024)) + " GB";
}

// 编译器屏障，零运行时开销。原理详见 test_final_fair.cpp。
// volatile: 不能删除/重排; "g"(p): p 及其上游计算不能被消除;
// "memory": 所有内存写入必须落地，阻止编译器缓存优化。
void doNotOptimize(void* p) {
    asm volatile("" : : "g"(p) : "memory");
}

int main() {
    // 测试尺寸从 1 cache line (64B) 到远超 L3 (256MB)，覆盖各级缓存边界
    size_t sizes[] = {
        64,                     // 1 cache line
        256,                    // 4 cache lines
        1024,                   // 1 KB
        4 * 1024,               // 4 KB (page)
        32 * 1024,              // 32 KB  (~L1d)
        40 * 1024,              // 40 KB  (原实验值)
        256 * 1024,             // 256 KB (~L2)
        1024 * 1024,            // 1 MB
        4 * 1024 * 1024,        // 4 MB   (~L3)
        16 * 1024 * 1024,       // 16 MB  (超出 L3)
        64 * 1024 * 1024,       // 64 MB
        256 * 1024 * 1024ULL,   // 256 MB
    };

    cout << fixed << setprecision(3);
    cout << left
         << setw(12) << "大小"
         << setw(10) << "轮次"
         << right
         << setw(14) << "memset"
         << setw(14) << "memcpy"
         << setw(12) << "差距"
         << endl;
    cout << string(62, '-') << endl;

    for (auto sz : sizes) {
        // 预分配并 touch 所有页面，排除 page fault 干扰
        // dst 填非零值，确保物理页已映射且不是 CoW 零页
        char* dst = (char*)malloc(sz);
        memset(dst, 1, sz);
        doNotOptimize(dst);

        // src 作为 memcpy 的源 buffer
        char* src = (char*)malloc(sz);
        memset(src, 0, sz);
        doNotOptimize(src);

        // 自适应迭代次数：小 buffer 多跑摊薄计时精度误差，大 buffer 少跑避免太慢
        int iters;
        if (sz <= 4096)              iters = 1000000;
        else if (sz <= 256 * 1024)   iters = 100000;
        else if (sz <= 4 * 1024 * 1024) iters = 10000;
        else if (sz <= 64 * 1024 * 1024) iters = 1000;
        else                          iters = 100;

        // warmup: 稳定分支预测、TLB、IFUNC 分发
        int warmup = iters / 10;
        if (warmup < 3) warmup = 3;
        for (int i = 0; i < warmup; i++) {
            memset(dst, 0, sz);
            doNotOptimize(dst);
        }
        for (int i = 0; i < warmup; i++) {
            memcpy(dst, src, sz);
            doNotOptimize(dst);
        }

        // memset
        auto t0 = chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            memset(dst, 0, sz);
            doNotOptimize(dst);
        }
        auto t1 = chrono::high_resolution_clock::now();

        // memcpy
        auto t2 = chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            memcpy(dst, src, sz);
            doNotOptimize(dst);
        }
        auto t3 = chrono::high_resolution_clock::now();

        double ms_set = chrono::duration<double, milli>(t1 - t0).count();
        double ms_cpy = chrono::duration<double, milli>(t3 - t2).count();
        double pct = (ms_cpy - ms_set) / ms_set * 100.0;

        cout << left << setw(12) << format_size(sz)
             << setw(10) << iters
             << right
             << setw(10) << ms_set << " ms"
             << setw(10) << ms_cpy << " ms"
             << setw(8) << showpos << pct << noshowpos << "%"
             << endl;

        free(dst);
        free(src);
    }

    return 0;
}
