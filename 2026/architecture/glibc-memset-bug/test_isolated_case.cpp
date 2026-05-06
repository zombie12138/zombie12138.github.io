// test_isolated_case.cpp — 原始问题复现：vector<vector<int>> 三种构造方式对比
//
// 目的：对比 fill constructor / resize loop / flat vector 的构造耗时，
//       同时记录 page fault 数量，用于验证测试顺序是否影响结果。
//
// 三种方式的底层差异：
//   A (fill constructor): 创建 prototype 行，循环 memcpy → 读+写
//   B (resize loop):      每行独立 memset 零初始化 → 纯写
//   C (flat vector):      单次大 memset → 纯写，且内存连续无碎片
//
// 用法：./test_isolated_case <rows> <cols>
// 编译：g++ -O3 -o test_isolated_case test_isolated_case.cpp

#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <cstdio>

using namespace std;

// 编译器屏障，零运行时开销。原理详见 test_final_fair.cpp。
// volatile: 不能删除/重排; "g"(p): p 及其上游计算不能被消除;
// "memory": 所有内存写入必须落地，阻止编译器缓存优化。
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

// A: fill constructor — vector(n, prototype) 内部对每行 memcpy prototype
double test_constructor(int rows, int cols, int rounds) {
    double total = 0;
    for (int r = 0; r < rounds; r++) {
        auto start = chrono::high_resolution_clock::now();
        vector<vector<int>> mat(rows, vector<int>(cols));
        doNotOptimize(&mat);
        auto end = chrono::high_resolution_clock::now();
        total += chrono::duration<double, milli>(end - start).count();
    }
    return total / rounds;
}

// B: resize loop — 每行独立 resize，走 _M_default_append → memset
double test_resize_loop(int rows, int cols, int rounds) {
    double total = 0;
    for (int r = 0; r < rounds; r++) {
        auto start = chrono::high_resolution_clock::now();
        vector<vector<int>> mat(rows);
        for (auto& row : mat) row.resize(cols);
        doNotOptimize(&mat);
        auto end = chrono::high_resolution_clock::now();
        total += chrono::duration<double, milli>(end - start).count();
    }
    return total / rounds;
}

// C: flat vector — 单块连续内存，单次 memset，作为性能上界参考
double test_flat_vector(int rows, int cols, int rounds) {
    double total = 0;
    for (int r = 0; r < rounds; r++) {
        auto start = chrono::high_resolution_clock::now();
        vector<int> mat((size_t)rows * cols);
        doNotOptimize(&mat);
        auto end = chrono::high_resolution_clock::now();
        total += chrono::duration<double, milli>(end - start).count();
    }
    return total / rounds;
}

string format_size(size_t bytes) {
    if (bytes < 1024) return to_string(bytes) + " B";
    if (bytes < 1024 * 1024) return to_string(bytes / 1024) + " KB";
    if (bytes < 1024ULL * 1024 * 1024) return to_string(bytes / (1024 * 1024)) + " MB";
    return to_string(bytes / (1024ULL * 1024 * 1024)) + " GB";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <rows> <cols>\n", argv[0]);
        return 1;
    }
    int rows = atoi(argv[1]);
    int cols = atoi(argv[2]);
    size_t total_bytes = (size_t)rows * cols * sizeof(int);

    // 根据数据量调整迭代次数：小矩阵多跑取平均，大矩阵少跑避免太慢
    int rounds;
    if (total_bytes < 1024 * 1024)            rounds = 1000;
    else if (total_bytes < 100ULL * 1024 * 1024) rounds = 10;
    else                                       rounds = 3;

    // 在每组测试前后采样 page fault 计数，用于验证：
    // 1. 各方式的 page fault 数量是否一致（排除顺序干扰）
    // 2. glibc heap trimming 是否让每次迭代都重新 fault
    long pf0 = read_minflt();
    double t1 = test_constructor(rows, cols, rounds);
    long pf1 = read_minflt();
    double t2 = test_resize_loop(rows, cols, rounds);
    long pf2 = read_minflt();
    double t3 = test_flat_vector(rows, cols, rounds);
    long pf3 = read_minflt();

    string dim = to_string(rows) + "x" + to_string(cols);
    cout << fixed << setprecision(3);
    cout << left << setw(18) << dim
         << setw(10) << format_size(total_bytes)
         << setw(6) << rounds
         << right
         << setw(10) << t1 << " ms  "
         << setw(10) << t2 << " ms  "
         << setw(10) << t3 << " ms"
         << "  | PF: A=" << (pf1-pf0) << " B=" << (pf2-pf1) << " C=" << (pf3-pf2)
         << left << endl;
    return 0;
}
