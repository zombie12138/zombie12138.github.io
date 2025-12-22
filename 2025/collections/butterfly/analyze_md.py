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
API_KEY = "APIKEY"  # 替换为你的 DeepSeek API Key
BASE_URL = "https://api.deepseek.com"       # DeepSeek Base URL
TARGET_DIR = "./"                           # 目标文件夹，默认当前目录
FILE_PATTERN = "source/_posts/**/*.md"                    # 递归查找所有 md 文件
OUTPUT_FILE = "summary_report.md"           # 汇总输出文件
MAX_WORKERS = 20                             # 并发线程数（根据你的 Rate Limit 调整）
# ===========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def parse_front_matter(content):
    """
    使用正则提取 YAML Front Matter 中的 tags。
    支持 list 格式 (- item) 和 inline 格式 ([item, item])
    """
    tags = []
    # 匹配 --- 之间的内容
    fm_match = re.search(r'^---\s+(.*?)\s+---', content, re.DOTALL)
    if fm_match:
        fm_content = fm_match.group(1)
        
        # 寻找 tags: 字段
        # 匹配 tags: 后面跟随的内容，直到遇到换行且下一行看起来像是一个新的 key (即不缩进)
        tags_match = re.search(r'^tags:\s*(.*?)(?=\n\w+:|\Z)', fm_content, re.MULTILINE | re.DOTALL)
        
        if tags_match:
            raw_tags = tags_match.group(1).strip()
            
            # 情况 1: tags: [hexo, theme]
            if raw_tags.startswith('['):
                clean_tags = raw_tags.strip("[]")
                tags = [t.strip() for t in clean_tags.split(',')]
            # 情况 2: tags: \n - hexo \n - theme
            else:
                # 寻找所有以 - 开头的行
                list_items = re.findall(r'^\s*-\s+(.*)', raw_tags, re.MULTILINE)
                tags = [t.strip() for t in list_items]
                
                # 如果没找到 list 且 raw_tags 不为空，可能是 tags: tag1 (单行无 list)
                if not tags and raw_tags:
                    tags = [raw_tags]

    return tags

def get_ai_analysis(content, filename):
    """
    调用 DeepSeek API 获取总结和打分
    """
    # 截取过长的文本以防止 Token 溢出 (保留前 4000 字符通常足够做摘要)
    truncated_content = content[:60000]
    
    system_prompt = """
## 基础要求

你是一个**专业、客观、中立的第三方技术博客归档与评审助手**。
请分析用户输入的 Markdown 内容，并**仅以严格的 JSON 格式输出结果**，不得包含 Markdown 代码块标记或任何额外说明文字。

输出必须包含以下字段：

1. `"summary"`
   用不超过两句话概括文章的核心技术内容，不允许评价性语言。

2. `"score_detail"`
   按以下维度分别评分，取值范围为 0–10，允许一位小数：

   * `technical_depth`：技术原理、源码级分析、推导深度
   * `originality`：是否包含原创结论、独立分析或非平凡经验
   * `completeness`：问题背景、过程、结论是否闭环
   * `practical_value`：对工程实践或技术决策的可复用价值
   * `workload_signal`：有效技术投入强度（不等于篇幅）

3. `"score"`
   按 **固定权重公式** 计算得到的最终分数（1–10，允许小数），禁止主观修正。

4. `"技术栈和工具"`
   列出文中实际使用或深入分析的技术栈与工具，并给出**简要技术定位说明**，不得因数量或流行度抬高评分。

5. `"优点"`
   列出 1–3 条，每条一句话，必须基于具体内容。

6. `"缺点"`
   列出 1–3 条，每条一句话，指出真实技术或表达层面的不足，不得使用泛化措辞。

---

## 评分规则（强约束）

### 一、评分分布约束（必须遵守）

假设你正在评审一个**长期维护、文章数量较多的技术博客**，其评分分布满足：

* **7.0 左右为主流**
* **8.0 以上不超过 30%**
* **9.0 以上不超过 10%**

当前文章的评分**必须服从该长期分布预期**。
评分采用**严格相对标准**，不是鼓励性评分。

---

### 二、评分区间语义（不可扩展解释）

* **9.0–10.0**
  可作为稳定技术参考资料，被他人引用或复现，包含明确、可迁移的技术结论，极少出现。

* **8.0–8.9**
  系统性技术分析，包含实验、对照或源码级推导，但影响范围有限。

* **7.0–7.9**
  完整的问题分析或经验总结，对读者有启发，但技术深度或结论不可外推。

* **6.0–6.9**
  实践记录、配置说明或使用笔记，技术增量较小。

* **≤5.9**
  流水账、主观记录或非技术性内容。

如果无法明确支撑高分区间语义，**必须下调评分**。

---

### 三、评分计算方式（硬性公式）

最终 `"score"` **必须且只能**由 `"score_detail"` 按以下权重计算得出：

* `technical_depth`：30%
* `originality`：25%
* `practical_value`：20%
* `completeness`：15%
* `workload_signal`：10%

禁止基于整体印象、写作质量或个人偏好调整最终分数。

**硬限制：**

* 任一单项 ≤ 6.0，则最终 score **不得超过 7.5**。

---

### 四、workload_signal 负面清单（必须遵守）

`workload_signal` **不得**因以下因素提高评分：

* 文章篇幅或行数
* 日志式过程记录
* 环境搭建步骤堆叠
* 配置参数或命令罗列
* 第三方文档的整理或翻译

仅认可以下信号：

* 实验设计与变量控制
* 失败路径与对照分析
* 技术取舍与权衡
* 原因归因或机制解释

---

### 五、额外压分假设（默认成立）

在评分时，**必须假设当前文章并非该博客中最优秀的少数**。
只有在出现**明确可复用的技术结论、方法论或源码级洞察**时，才允许进入 8.5 以上区间。

---

### 六、输出格式

```json
{
  "summary": "总结文字在这里",
  "score_detail": {
    "technical_depth": 0.0,
    "originality": 0.0,
    "completeness": 0.0,
    "practical_value": 0.0,
    "workload_signal": 0.0
  },
  "score": 0.0,
  "技术栈和工具": ["工具1","工具2","工具3"],
  "优点": ["优点1"],
  "缺点": ["缺点1"]
}

```
"""
    
    user_prompt = f"文件名: {filename}\n\n文件内容摘要:\n{truncated_content}"

    try:
        response = client.chat.completions.create(
            model="deepseek-chat", # 或者 deepseek-coder
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"} # 强制 JSON 模式
        )
        result = response.choices[0].message.content
        return json.loads(result)
    except Exception as e:
        return {"summary": f"分析失败: {str(e)}", "score": 0}

def process_file(file_path):
    """
    单个文件处理逻辑：读取 -> Python分析 -> AI分析
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 1. Python 统计行数 (比 AI 数得准)
        line_count = len(content.splitlines())
        
        # 2. Python 提取 Tags
        tags = parse_front_matter(content)
        
        # 3. AI 分析内容
        # 如果文件太短（例如只有 front matter），跳过 AI 以节省 token
        if len(content) < 50:
            ai_result = {"summary": "内容过短，未进行 AI 分析。", "score": 0}
        else:
            ai_result = get_ai_analysis(content, os.path.basename(file_path))
            
        return {
            "status": "success",
            "path": file_path,
            "filename": os.path.basename(file_path),
            "lines": line_count,
            "tags": tags,
            "ai_result": ai_result
        }
        
    except Exception as e:
        return {
            "status": "error",
            "path": file_path,
            "filename": os.path.basename(file_path),
            "error": str(e)
        }

def main():
    # 获取所有 md 文件
    files = glob.glob(os.path.join(TARGET_DIR, FILE_PATTERN), recursive=True)
    # 过滤掉输出文件本身，防止死循环
    files = [f for f in files if os.path.basename(f) != OUTPUT_FILE]
    
    print(f"找到 {len(files)} 个 Markdown 文件，开始并行分析...")
    print(f"使用模型: deepseek-chat, 并发数: {MAX_WORKERS}")

    results = []
    
    # 并行执行
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任务
        future_to_file = {executor.submit(process_file, f): f for f in files}
        
        # 使用 tqdm 显示进度条
        for future in tqdm(as_completed(future_to_file), total=len(files), unit="file"):
            results.append(future.result())

    # 按分数降序排序 (高质量文章排前面)
    results.sort(key=lambda x: x.get("score", 0) if x.get("status") == "success" else -1, reverse=True)

    # 写入汇总报告
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"# DeepSeek Markdown 归档分析报告\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"文件总数: {len(files)}\n\n")
        f.write("---\n\n")

        for res in results:
            if res['status'] == 'error':
                f.write(f"### ❌ {res['filename']}\n")
                f.write(f"> 读取或处理出错: {res['error']}\n\n")
                continue

            # 格式化 Tags
            tags_str = ", ".join([f"`{t}`" for t in res['tags']]) if res['tags'] else "无"
            
            ai = res["ai_result"]

            f.write(f"### {res['filename']}\n")
            f.write(f"- **行数**: {res['lines']} lines\n")
            f.write(f"- **Tags**: {tags_str}\n\n")

            f.write("#### AI 评审\n\n")

            f.write("- **摘要**\n")
            f.write(f"  {ai['summary']}\n\n")

            f.write(f"- **评分**: **{ai['score']} / 10**\n\n")

            f.write("- **评分明细**\n")
            for k, v in ai["score_detail"].items():
                f.write(f"  - {k}: {v}\n")
            f.write("\n")

            f.write("- **技术栈和工具**\n")
            for item in ai["技术栈和工具"]:
                f.write(f"  - {item}\n")
            f.write("\n")

            f.write("- **优点**\n")
            for item in ai["优点"]:
                f.write(f"  - {item}\n")
            f.write("\n")

            f.write("- **缺点**\n")
            for item in ai["缺点"]:
                f.write(f"  - {item}\n")
            f.write("\n---\n\n")
            f.write("---\n\n")

    print(f"\n✅ 分析完成！汇总报告已生成至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()