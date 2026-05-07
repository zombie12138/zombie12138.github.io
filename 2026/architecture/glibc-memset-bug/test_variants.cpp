// test_variants.cpp — vector<vector<int>> 五种构造方式性能对比
//
// 目的：验证不同 STL 构造方式（fill constructor / resize / assign /
//       emplace_back / move assignment）在底层走的代码路径差异，以及
//       这些差异对性能的影响。
//
// 核心发现：
//   - A (fill constructor) 走 memcpy（复制 prototype 行）
//   - B1/B2/B3/B4 最终都走 memset（零初始化新内存）
//   - 差异来源是 memcpy vs memset，而非 STL 接口本身的开销
//
// 编译：g++ -O3 -o test_variants test_variants.cpp

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>

using namespace std;

// 编译器屏障，零运行时开销。原理详见 test_final_fair.cpp。
// volatile: 不能删除/重排; "g"(p): p 及其上游计算不能被消除;
// "memory": 所有内存写入必须落地，阻止编译器缓存优化。
void doNotOptimize(void* p) {
    asm volatile("" : : "g"(p) : "memory");
}

// A: fill constructor — vector(n, prototype)，内部循环 memcpy prototype 到每一行
double test_A(int rows, int cols, int rounds) {
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

// B1: resize — 每行调 resize(cols)，走 _M_default_append → memset 零初始化
double test_B1(int rows, int cols, int rounds) {
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

// B2: assign — 每行调 assign(cols, 0)，语义是"用 n 个值填充"，底层也走 memset
double test_B2(int rows, int cols, int rounds) {
    double total = 0;
    for (int r = 0; r < rounds; r++) {
        auto start = chrono::high_resolution_clock::now();
        vector<vector<int>> mat(rows);
        for (auto& row : mat) row.assign(cols, 0);
        doNotOptimize(&mat);
        auto end = chrono::high_resolution_clock::now();
        total += chrono::duration<double, milli>(end - start).count();
    }
    return total / rounds;
}

// B3: emplace_back — 先 reserve 再逐行 emplace_back(cols, 0)，每行原地构造
double test_B3(int rows, int cols, int rounds) {
    double total = 0;
    for (int r = 0; r < rounds; r++) {
        auto start = chrono::high_resolution_clock::now();
        vector<vector<int>> mat;
        mat.reserve(rows);
        for (int i = 0; i < rows; i++)
            mat.emplace_back(cols, 0);
        doNotOptimize(&mat);
        auto end = chrono::high_resolution_clock::now();
        total += chrono::duration<double, milli>(end - start).count();
    }
    return total / rounds;
}

// B4: move assignment — 每行构造临时 vector 后 move 赋值，验证 move 语义是否有额外开销
double test_B4(int rows, int cols, int rounds) {
    double total = 0;
    for (int r = 0; r < rounds; r++) {
        auto start = chrono::high_resolution_clock::now();
        vector<vector<int>> mat(rows);
        for (auto& row : mat) row = vector<int>(cols);
        doNotOptimize(&mat);
        auto end = chrono::high_resolution_clock::now();
        total += chrono::duration<double, milli>(end - start).count();
    }
    return total / rounds;
}

// C1: flat vector + mdspan — 单块连续内存，C++23 多维视图，零开销抽象
// 编译需要 -std=c++23，且编译器支持 <mdspan>（GCC 14 需 libstdc++ trunk 或用 Kokkos mdspan）
#if __has_include(<mdspan>)
#include <mdspan>
#define HAS_MDSPAN 1
#else
#define HAS_MDSPAN 0
#endif

#if HAS_MDSPAN
double test_C1_mdspan(int rows, int cols, int rounds) {
    double total = 0;
    for (int r = 0; r < rounds; r++) {
        auto start = chrono::high_resolution_clock::now();
        vector<int> storage(static_cast<size_t>(rows) * cols);
        auto mat = std::mdspan(storage.data(),
                               std::extents<int, std::dynamic_extent, std::dynamic_extent>{rows, cols});
        doNotOptimize(&storage);
        doNotOptimize(&mat);
        auto end = chrono::high_resolution_clock::now();
        total += chrono::duration<double, milli>(end - start).count();
    }
    return total / rounds;
}
#endif

// C2: flat vector + views::chunk — 单块连续内存，C++23 range 视图分块
#if __cpp_lib_ranges_chunk >= 202202L
#define HAS_CHUNK 1
#else
#define HAS_CHUNK 0
#endif

#if HAS_CHUNK
#include <ranges>
double test_C2_chunk(int rows, int cols, int rounds) {
    double total = 0;
    for (int r = 0; r < rounds; r++) {
        auto start = chrono::high_resolution_clock::now();
        vector<int> storage(static_cast<size_t>(rows) * cols);
        auto mat = storage | std::views::chunk(cols);
        doNotOptimize(&storage);
        doNotOptimize(&mat);
        auto end = chrono::high_resolution_clock::now();
        total += chrono::duration<double, milli>(end - start).count();
    }
    return total / rounds;
}
#endif

// C0: flat vector — 作为 C1/C2 的基准对照
double test_C0_flat(int rows, int cols, int rounds) {
    double total = 0;
    for (int r = 0; r < rounds; r++) {
        auto start = chrono::high_resolution_clock::now();
        vector<int> mat(static_cast<size_t>(rows) * cols);
        doNotOptimize(&mat);
        auto end = chrono::high_resolution_clock::now();
        total += chrono::duration<double, milli>(end - start).count();
    }
    return total / rounds;
}

int main() {
    struct Case { int rows; int cols; };
    Case cases[] = {{1000, 10}, {100000, 10}, {100000, 100}};

    cout << fixed << setprecision(3);
    cout << left << setw(16) << "维度"
         << right
         << setw(12) << "A(fill)"
         << setw(16) << "B1(resize)"
         << setw(16) << "B2(assign)"
         << setw(18) << "B3(emplace_back)"
         << setw(16) << "B4(move=)"
         << setw(12) << "C0(flat)";
#if HAS_MDSPAN
    cout << setw(14) << "C1(mdspan)";
#endif
#if HAS_CHUNK
    cout << setw(14) << "C2(chunk)";
#endif
    cout << endl;

    int total_cols = 106;
#if HAS_MDSPAN
    total_cols += 14;
#endif
#if HAS_CHUNK
    total_cols += 14;
#endif
    cout << string(total_cols, '-') << endl;

    for (auto& c : cases) {
        int rounds = 1000;
        string dim = to_string(c.rows) + "x" + to_string(c.cols);

        double a  = test_A(c.rows, c.cols, rounds);
        double b1 = test_B1(c.rows, c.cols, rounds);
        double b2 = test_B2(c.rows, c.cols, rounds);
        double b3 = test_B3(c.rows, c.cols, rounds);
        double b4 = test_B4(c.rows, c.cols, rounds);
        double c0 = test_C0_flat(c.rows, c.cols, rounds);

        cout << left << setw(16) << dim << right
             << setw(9) << a << " ms"
             << setw(13) << b1 << " ms"
             << setw(13) << b2 << " ms"
             << setw(15) << b3 << " ms"
             << setw(13) << b4 << " ms"
             << setw(9) << c0 << " ms";
#if HAS_MDSPAN
        double c1 = test_C1_mdspan(c.rows, c.cols, rounds);
        cout << setw(11) << c1 << " ms";
#endif
#if HAS_CHUNK
        double c2 = test_C2_chunk(c.rows, c.cols, rounds);
        cout << setw(11) << c2 << " ms";
#endif
        cout << endl;
    }
    return 0;
}
