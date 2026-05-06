// test_final_fair.cpp — 公平对比 memset(0) / memset(1) / memcpy 的单次操作耗时
//
// 目的：验证 memset(0) 在 glibc 中是否走了不同的代码路径（non-temporal store），
//       以及 page fault 对测量结果的干扰程度。
//
// 设计：每次迭代都 malloc 一块新的 64MB buffer，可选是否预 touch（prefault）来
//       隔离 page fault 的影响。每次迭代后 free，glibc 会 brk() 收缩归还物理页，
//       保证下一次迭代的 page fault 行为一致。
//
// 用法：./test_final_fair <memset0|memset1|memcpy> <prefault|noprefault>
//   - memset0 vs memset1 对比：揭示 glibc memset(0) 的特殊路径
//   - prefault vs noprefault 对比：量化 page fault 的影响
//
// 编译：g++ -O3 -o test_final_fair test_final_fair.cpp

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <chrono>

using namespace std;

// 编译器屏障，等价于 Google Benchmark 的 DoNotOptimize。
// asm volatile: 告诉编译器这是有副作用的内联汇编，不能删除也不能重排。
// "g"(p): 将 p 声明为输入操作数，编译器必须保证 p（及产生 p 的计算）在此处有效，
//         不能将上游的 malloc/memset 等操作当作死代码消除。
// "memory": clobber 声明——告诉编译器此 asm 块可能读写任意内存，
//           编译器必须在此前将所有 pending store 刷到内存（不能只保留寄存器副本），
//           且之后不能假设任何内存值仍然有效。
// 三者缺一不可：没有 "g"(p) → p 的产生过程可能被消除；
//               没有 "memory" → p 指向的实际数据可能不被写入；
//               没有 volatile → 整个 asm 块可能被删除。
// 实际生成零条机器指令，纯编译期约束，运行时零开销。
void doNotOptimize(void* p) {
    asm volatile("" : : "g"(p) : "memory");
}

// 从 /proc/self/stat 读取 minor page fault 计数（第 10 个字段）
long read_minflt() {
    FILE* f = fopen("/proc/self/stat", "r");
    long minflt;
    fscanf(f, "%*d %*s %*c %*d %*d %*d %*d %*d %*u %lu", &minflt);
    fclose(f);
    return minflt;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <op> <mode>\n", argv[0]);
        printf("  op: memset0, memset1, memcpy\n");
        printf("  mode: prefault, noprefault\n");
        return 1;
    }

    size_t sz = 64ULL * 1024 * 1024;  // 64 MB
    int iters = 100;

    // memcpy 模式下需要一个源 buffer，填充非零值避免 CoW 零页优化
    char* src = NULL;
    if (strcmp(argv[1], "memcpy") == 0) {
        src = (char*)malloc(sz);
        memset(src, 0x42, sz);
        doNotOptimize(src);
    }

    int prefault = strcmp(argv[2], "prefault") == 0;
    double total = 0;
    long total_pf = 0;

    for (int i = 0; i < iters; i++) {
        char* dst = (char*)malloc(sz);

        // prefault 模式：先写一遍触发所有 page fault，使后续操作在已映射的页上进行
        if (prefault) {
            memset(dst, 0xAA, sz);
            doNotOptimize(dst);
        }

        long pf0 = read_minflt();
        auto t0 = chrono::high_resolution_clock::now();

        if (strcmp(argv[1], "memset0") == 0)
            memset(dst, 0, sz);
        else if (strcmp(argv[1], "memset1") == 0)
            memset(dst, 1, sz);
        else
            memcpy(dst, src, sz);
        doNotOptimize(dst);

        auto t1 = chrono::high_resolution_clock::now();
        total += chrono::duration<double, milli>(t1 - t0).count();
        total_pf += read_minflt() - pf0;

        // free 后 glibc 调用 brk() 收缩堆，内核回收物理页，
        // 保证下一次 malloc 的 page fault 行为与第一次一致
        free(dst);
    }

    printf("%-8s %-10s avg=%.3f ms  pf=%ld\n",
           argv[1], argv[2], total / iters, total_pf);
    return 0;
}
