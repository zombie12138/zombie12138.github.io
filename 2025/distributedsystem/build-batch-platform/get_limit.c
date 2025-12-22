#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>

// 读取指定路径的整数值
long read_integer_from_file(const char *path) {
    FILE *file = fopen(path, "r");
    if (!file) {
        perror("Failed to open file");
        return -1;
    }

    long value = -1;
    if (fscanf(file, "%ld", &value) != 1) {
        perror("Failed to read value from file");
    }

    fclose(file);
    return value;
}

// 读取 /sys/fs/cgroup/memory 文件 (cgroup v1 或 v2)
long read_cgroup_memory_limit() {
    long limit = -1;

    // 读取 cgroup v2 的 memory.max（如果存在）
    limit = read_integer_from_file("/sys/fs/cgroup/memory.max");
    if (limit != -1) {
        return limit;
    }

    // 如果 v2 文件未找到，尝试读取 cgroup v1 的 memory.limit_in_bytes
    return read_integer_from_file("/sys/fs/cgroup/memory/memory.limit_in_bytes");
}

// 读取 CPU 配额，cgroup v2 或 v1
long read_cgroup_cpu_limit() {
    long limit = -1;

    // 尝试读取 cgroup v2 的 cpu.max
    limit = read_integer_from_file("/sys/fs/cgroup/cpu.max");
    if (limit != -1) {
        return limit;
    }

    // 尝试读取 cgroup v1 的 cpu.cfs_quota_us 和 cpu.cfs_period_us
    long quota = read_integer_from_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us");
    long period = read_integer_from_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
    if (quota > 0 && period > 0) {
        return quota / period;
    }

    return limit;
}

// 读取 CPU 核心数
int get_cpu_cores() {
    return sysconf(_SC_NPROCESSORS_ONLN);
}

// 读取物理内存总量（单位：字节）
long get_total_memory() {
    struct sysinfo info;
    if (sysinfo(&info) != 0) {
        perror("sysinfo failed");
        return -1;
    }
    return info.totalram;
}

// 读取进程最大资源限制（例如：最大虚拟内存）
long get_rlimit_max(int resource) {
    struct rlimit rlim;
    if (getrlimit(resource, &rlim) == 0) {
        return rlim.rlim_cur;
    }
    return -1;
}

int main() {
    // 显示容器的内存和 CPU 限制
    printf("=== CGroup Resource Limits ===\n");

    long cgroup_mem_limit = read_cgroup_memory_limit();
    if (cgroup_mem_limit != -1) {
        printf("CGroup Memory Limit: %ld bytes\n", cgroup_mem_limit);
    } else {
        printf("CGroup Memory Limit: Not found (maybe not set)\n");
    }

    long cgroup_cpu_limit = read_cgroup_cpu_limit();
    if (cgroup_cpu_limit != -1) {
        printf("CGroup CPU Limit: %ld CPU shares\n", cgroup_cpu_limit);
    } else {
        printf("CGroup CPU Limit: Not found (maybe not set)\n");
    }

    // 显示 Linux 本身的资源限制
    printf("\n=== Linux System Resource Limits ===\n");

    long total_memory = get_total_memory();
    if (total_memory != -1) {
        printf("Total Physical Memory: %ld bytes\n", total_memory);
    }

    printf("CPU Cores: %d\n", get_cpu_cores());

    // 最大虚拟内存限制
    long max_virtual_memory = get_rlimit_max(RLIMIT_AS);
    if (max_virtual_memory != -1) {
        printf("Max Virtual Memory (RLIMIT_AS): %ld bytes\n", max_virtual_memory);
    }

    // 最大文件描述符数限制
    long max_open_files = get_rlimit_max(RLIMIT_NOFILE);
    if (max_open_files != -1) {
        printf("Max Open Files (RLIMIT_NOFILE): %ld\n", max_open_files);
    }

    return 0;
}
