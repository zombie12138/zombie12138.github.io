# -*- coding: UTF-8 -*-
# /***********************************************
# Copyright (c) 2019, Shanghai
# All rights reserved.
#
# @Filename: link.py
# @Version：V1.0
# @Author: Frank Liu - frankliu624@gmail.com
# @Description: ---
# @Create Time: 2025-09-08 23:20:03
# @Last Modified: 2025-09-08 23:20:03
# ***********************************************/
# 为什么 BFS（GPT补充），以及 first found 文档支持。
import os
import subprocess
from textwrap import dedent

# 定义依赖树
tree = {
    "A": ["A1", "A2"],
    "B": ["B1", "B2"],
    "A1": ["A1a", "A1b"],
    "A2": ["A2a", "A2b"],
    "B1": ["B1a", "B1b"],
    "B2": ["B2a", "B2b"],
    # 叶子节点
    "A1a": [], "A1b": [], "A2a": [], "A2b": [],
    "B1a": [], "B1b": [], "B2a": [], "B2b": []
}

def gen_leaf(name):
    return dedent(f"""
    #include <stdio.h>
    void hello_{name}() {{
        printf("hello from {name}\\n");
    }}
    """)

def gen_middle(name, children):
    decls = "\n".join([f"void hello_{c}();" for c in children])
    calls = "\n    ".join([f"hello_{c}();" for c in children])
    return dedent(f"""
    #include <stdio.h>
    {decls}

    void hello_{name}() {{
        printf("hello from {name}\\n");
        {calls}
    }}
    """)

def gen_app():
    return dedent("""
    #include <stdio.h>
    void hello_A();
    void hello_B();
    int main() {
        printf("hello from app\\n");
        hello_A();
        hello_B();
        return 0;
    }
    """)

def write_sources():
    with open("app.c", "w") as f:
        f.write(gen_app())
    for name, children in tree.items():
        if children:
            code = gen_middle(name, children)
        else:
            code = gen_leaf(name)
        with open(f"lib{name}.c", "w") as f:
            f.write(code)

def run(cmd, env=None):
    print(f"$ {' '.join(cmd)}")
    subprocess.check_call(cmd, env=env)

def build():
    # 清理
    for f in os.listdir("."):
        if f.endswith(".so") or f.endswith(".c") or f == "app":
            os.remove(f)

    write_sources()

    # 叶子
    for name, children in tree.items():
        if not children:
            run(["gcc", "-fPIC", "-shared", "-o", f"lib{name}.so", f"lib{name}.c"])

    # 按层编译
    levels = [["A1a","A1b","A2a","A2b","B1a","B1b","B2a","B2b"],
              ["A1","A2","B1","B2"],
              ["A","B"]]
    for level in levels[1:]:
        for name in level:
            libs = [f"-l{c}" for c in tree[name]]
            run(["gcc", "-fPIC", "-shared", "-o", f"lib{name}.so", f"lib{name}.c", "-L."] + libs)

    # 主程序
    run(["gcc", "-o", "app", "app.c", "-L.", "-lA", "-lB", "-Wl,-rpath=."])

def run_app():
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "."
    env["LD_DEBUG"] = "libs"
    run(["./app"], env=env)

if __name__ == "__main__":
    build()
    run_app() 
