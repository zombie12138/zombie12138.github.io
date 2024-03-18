#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

void *func_addr = NULL;
size_t page_size = 0;
void segv_handler(int sig, siginfo_t *si, void *unused) {
  // 检查触发SIGSEGV的地址是否为我们关注的函数地址
  if (si->si_addr == func_addr) {
    printf("Attempt to execute function at %p caught\n", si->si_addr);

    // 暂时恢复执行权限
    if (mprotect((void *)((unsigned long)func_addr & ~(page_size - 1)),
                 page_size, PROT_READ | PROT_EXEC) == -1) {
      perror("mprotect");
      exit(1);
    }
    // 这里可以执行函数或者做其他事情，然后再次移除执行权限
    // ...

    // 重新设置为不可执行
    if (mprotect((void *)((unsigned long)func_addr & ~(page_size - 1)),
                 page_size, PROT_READ) == -1) {
      perror("mprotect");
      exit(1);
    }
  }
}

#include <dlfcn.h>

int main(int argc, char *argv[]) {
  struct sigaction sa;
  void *handle;
  int (*example_function)(int, int);

  page_size = sysconf(_SC_PAGESIZE);

  // 设置信号处理函数
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_SIGINFO;
  sa.sa_sigaction = segv_handler;
  if (sigaction(SIGSEGV, &sa, NULL) == -1) {
    perror("sigaction");
    exit(1);
  }

  // 加载库并获取函数地址
  handle = dlopen("./libadd.so", RTLD_LAZY);
  if (!handle) {
    fprintf(stderr, "%s\n", dlerror());
    exit(1);
  }

  dlerror(); // 清除任何现有的错误

  *(void **)(&example_function) = dlsym(handle, "add");
  if ((func_addr = dlsym(handle, "add")) == NULL) {
    fprintf(stderr, "%s\n", dlerror());
    exit(1);
  }

  // 修改函数所在内存页的权限为不可执行
  if (mprotect((void *)((unsigned long)func_addr & ~(page_size - 1)), page_size,
               PROT_READ) == -1) {
    perror("mprotect");
    exit(1);
  }

  // 尝试执行函数（这将触发SIGSEGV并被我们的处理函数捕获）
  example_function(1, 2);

  dlclose(handle);
  return 0;
}

// gcc sigaction.c -o sig -g -L. -ladd -ldl