# pip install openai tqdm
import os
import glob
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
API_KEY = "sk-d8bfe936592748e0a23aadc339d1d041"                     # 替换为你的 DeepSeek API Key
BASE_URL = "https://api.deepseek.com"  # DeepSeek Base URL

TARGET_DIR = "./"
FILE_PATTERN = "source/_posts/**/*.md"  # 递归查找 md 文件
OUTPUT_FILE = "source/_posts/butterfly/post_tags.json"          # 输出 JSON
MAX_WORKERS = 40                        # 并发数（按 Rate Limit 调整）

MODEL = "deepseek-reasoner"                 # deepseek-chat / deepseek-coder / deepseek-reasoner 
TEMPERATURE = 0.1

MAX_CHARS = 400000                       # 传给模型的最大字符（避免超长）
RETRY = 3                               # API 重试次数
RETRY_BACKOFF_BASE = 1.8                # 退避系数
# ===========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

CORE_CATEGORIES = [
    "Language/C++",
    "Language/Java",
    "Language/Python",
    "Networking",
    "Systems",
    "Compiler",
    "Performance",
    "AI",
    "Devops",
]

DEPTH_GENRE_TAGS = [
    "DeepDive",
    "TroubleShooting",
    "Benchmark",
    "Note",
    "PaperReading",
    "Survey",
]

PROJECT_TAGS = [
    "Project/Abyss",
    "Project/Blog",
    "Project/HomeLab",
]

def strip_front_matter(md: str) -> str:
    # 去掉 YAML Front Matter，避免误导模型
    return re.sub(r"(?s)\A---\s.*?\s---\s*", "", md)

def safe_json_loads(s: str):
    # 兜底：有些模型会夹带前后空白/换行
    s = s.strip()
    return json.loads(s)

SYSTEM_PROMPT = """
你是一个【技术博客自动分类与标签生成器】。

你的任务是：
为一篇 Markdown 技术文章生成：
- **1 个 Category（必选，唯一）**
- **0–2 个 Intent**
- **0–4 个 Tech Tags**
- **0–1 个 Optional Tag**

你必须严格遵守下面的分类体系与判定规则。
你只能输出【严格 JSON】，不允许任何解释性文字、Markdown、注释或多余字段。

================================
一、Category（核心技术对象｜唯一）
================================

Category 只回答一个问题：

【这篇文章的“主叙事对象”是什么系统或技术域？】

注意：
- Category 是“户口”，不是全集
- 允许文章涉及多个技术域，但只能选择**主线叙事对象**
- 不追求互斥，只追求稳定和一致
- 禁止使用 Category 表达：视角、方法、目标、体裁、深度

允许值（只能选 1 个）：

- Kernel&OS  
  Linux Kernel、调度、虚拟内存、文件系统、并发原语、eBPF、虚拟化

- Architecture  
  CPU / GPU / 异构计算 / Memory Hierarchy / NUMA / 性能与软硬件协同

- DistributedSystem  
  一致性协议、分布式调度、分布式存储、K8s 控制面

- Networking  
  网络协议、通信机制、RPC、RDMA、SDN、QUIC

- PL&Compiler  
  编程语言、编译器、Runtime、LLVM、ABI、链接与加载

- AI  
  机器学习与算法本体：模型结构、训练方法、推理算法、RAG

- Database  
  数据库、缓存、消息队列、存储引擎

- Embedded  
  MCU / SoC / RTOS / 驱动 / IoT / 边缘系统
  
- Collections
  一些收藏，比如收藏的博客、名言；如果一些调研完全确认也没有合适分类，也可以放到这里，作为一种逃生通道

判定规则（非常重要）：
- 如果是 **AI 推理/训练系统工程、性能、并行、Runtime** → Architecture 或 DistributedSystem
- 如果是 **模型/算法/方法论本身** → AI
- 如果难以抉择，选择“文章主要展开分析篇幅最多的对象”

================================
二、Intent（文章意图｜0–2 个）
================================

Intent 描述文章在“做什么”，不是抽象层级，也不是技术对象。
Intent 允许组合，但最多 2 个。

允许值：

- Survey  
  生态调研、方案对比、论文综述、横向分析

- Theory  
  理论模型、算法分析、性能模型、形式化推导

- SourceCode  
  源码级分析、实现机制拆解

- Optimization  
  性能调优、瓶颈分析、优化路径（非目标函数）

- Benchmark  
  压测、对照实验、定量分析、数据报告

- TroubleShooting  
  故障排查闭环（现象 → 定位 → 修复 → 复盘）

- Design  
  系统设计、架构决策、Trade-off 分析

- Note  
  笔记、记录、清单、安装配置、备忘

规则：
- Intent 不是互斥事实，但必须是文章的“主行为”
- 如果只是顺带提及，不要选
- 不要为了凑满而选择

================================
三、Tech / Tags（技术实体｜0–4 个）
================================

Tech Tag 用于承载真实工程复杂度。

规则：
- 只允许**具体、可枚举的技术名词**
- 严禁使用 Category 同粒度或抽象词（如 System / Network / Performance）
- 只有当该技术是文章主线组成部分时才加入
- 数量上限 4，宁少勿滥

允许示例：

- 编程语言：Rust / Go / C++ / Python / TypeScript
- 组件工具：LLVM / Docker / Prometheus / ClickHouse
- 协议算法：Raft / gRPC / QUIC / io_uring
- 硬件实体：CPU / GPU / NUC / STM32F103

禁止示例：
- Debug / Performance / Security / Optimization
- Network / System / Backend

================================
四、Optional（上下文限定｜0–1 个）
================================

可选，用于表达“写作上下文”，不是技术事实。

- Project/XXX  
  例如：Project/Blog(仅仅涉及博客配置、写作方式、自动化写作等本身才可以使用)、Project/HomeLab、Project/Abyss

- Serial/XXX  
  持续专题，例如：Serial/JVM、Serial/LinuxMM

规则：
- 必须带前缀 Project/ 或 Serial/
- 最多 1 个

================================
四、可选形式（上下文限定｜0–1 个）
================================

- Reading
   阅读博客、论文

================================
六、组合与校验规则（强制）
================================

- Category：1 个（必须）
- Intent：0–2 个
- Tech：0–4 个
- Optional：0–1 个
- 总体遵循：**少而准确，不求覆盖一切**

================================
七、输出格式（严格）
================================

你只能输出如下 JSON 结构：

{
  "category": "Kernel&OS",
  "tags": ["SourceCode", "Optimization","eBPF", "LLVM", "CPU","Project/HomeLab"]
}

强制要求：
- 不允许输出任何解释、注释、Markdown
- 不允许缺字段时用 null，缺失就不输出该字段
- 不允许自造新值
- 选择结果必须可被人类长期一致复现
""".strip()

def call_ds_tagging(filename: str, content: str):
    # 内容截断：先去 front matter，再截断
    body = strip_front_matter(content)
    truncated = body[:MAX_CHARS]

    user_prompt = f"文件名: {filename}\n\nMarkdown 内容:\n{truncated}"

    last_err = None
    for i in range(RETRY):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
            )
            obj = safe_json_loads(resp.choices[0].message.content)

            # 本地强校验 + 规整
            category = obj.get("category", "")
            tags = obj.get("tags", [])

            # Project: 只能来自列表
            # projects = [t for t in tags if t in PROJECT_TAGS]
            # # Serial: 只能 Serial/ 前缀
            # serials = [t for t in tags if t.startswith("Serial/")]

            # # 重新按优先级组装并裁剪到 5
            # final_tags = []
            # final_tags.extend(depth[:1])
            # final_tags.extend(secondary[:3])
            # final_tags.extend(projects[:1])
            # final_tags.extend(serials[:1])
            # final_tags = final_tags[:5]

            return {"category": category, "tags": tags}

        except Exception as e:
            last_err = e
            sleep_s = (RETRY_BACKOFF_BASE ** i) + (0.05 * i)
            time.sleep(sleep_s)

    raise RuntimeError(f"Tagging failed after retries: {last_err}")

def process_file(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        if len(content.strip()) < 50:
            # 过短：用最低信息输出（仍满足 schema：体裁 + 1 个次要技术）
            # 次要技术允许自由命名，但这里无法从内容判断；用 "Misc" 会违反“不得自造(次要技术可自造)”但仍不可靠。
            # 因为规则允许次要技术自由命名，这里用 "Misc" 作为占位；如不想占位，直接报错更严格。
            return {
                "status": "success",
                "file": os.path.basename(path),
                "categories": ["Devops"],
                "tags": ["Note", "Misc"],
                "warning": "content too short; used placeholder secondary tech tag",
            }

        tagged = call_ds_tagging(os.path.basename(path), content)
        return {
            "status": "success",
            "file": os.path.basename(path),
            "categories": [tagged["category"]],
            "tags": tagged["tags"],
        }

    except Exception as e:
        return {
            "status": "error",
            "file": os.path.basename(path),
            "path": path,
            "error": str(e),
        }

def main():
    files = glob.glob(os.path.join(TARGET_DIR, FILE_PATTERN), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    files.sort()

    print(f"找到 {len(files)} 个 Markdown 文件，开始并行分类与打标...")
    print(f"模型: {MODEL} | 并发: {MAX_WORKERS}")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(process_file, p): p for p in files}
        for fut in tqdm(as_completed(futs), total=len(files), unit="file"):
            results.append(fut.result())

    # 分离错误
    ok = [r for r in results if r.get("status") == "success"]
    err = [r for r in results if r.get("status") == "error"]

    # 输出结构：只写 posts（错误另存）
    output = {"posts": [{"file": r["file"], "categories": r["categories"], "tags": r["tags"]} for r in ok]}

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    if err:
        with open("post_tags.errors.json", "w", encoding="utf-8") as f:
            json.dump({"errors": err}, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完成：成功 {len(ok)} / 失败 {len(err)}")
    print(f"输出: {OUTPUT_FILE}")
    if err:
        print("错误明细: post_tags.errors.json")

if __name__ == "__main__":
    main()
