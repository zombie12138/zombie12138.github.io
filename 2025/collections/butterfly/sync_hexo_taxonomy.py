#!/usr/bin/env python3
# sync_hexo_taxonomy_v2.py
# 1. 更新 tags/categories 到 Front-matter
# 2. 根据第一个 category (snake_case) 移动 .md 文件
# 3. [新增] 如果存在同名资源目录 (Asset Folder)，一并移动
#
# 目录规则：root/snake_case_category/filename.md
#          root/snake_case_category/filename/ (资源目录)

import argparse
import json
import os
import re
import sys
import shutil
from typing import Dict, List, Optional, Tuple

YAML_BOOL_NULL = {
    "y","yes","n","no","true","false","on","off","null","~"
}

def to_snake_case(text: str) -> str:
    """
    将字符串转换为 snake_case 目录名
    """
    if not text:
        return "uncategorized"
    
    s = str(text).lower().strip()
    s = s.replace("c++", "cpp").replace("c#", "csharp")
    s = re.sub(r'[\s\-\.]+', '_', s)
    s = re.sub(r'[^\w_]', '', s)
    s = s.strip('_')
    return s if s else "uncategorized"

def yaml_scalar(s: str) -> str:
    s = str(s)
    if s.strip() != s:
        return json.dumps(s, ensure_ascii=False)
    if re.fullmatch(r"[+-]?\d+(\.\d+)?([eE][+-]?\d+)?", s or ""):
        return json.dumps(s, ensure_ascii=False)
    if s.lower() in YAML_BOOL_NULL:
        return json.dumps(s, ensure_ascii=False)
    if re.fullmatch(r"[A-Za-z0-9_./+\-]+", s or ""):
        return s
    return json.dumps(s, ensure_ascii=False)

def load_mapping(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    posts = data.get("posts")
    if not isinstance(posts, list):
        raise ValueError("mapping json 缺少 posts 数组")
    return posts

def index_markdown_files(root: str) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = {}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(".md"):
                continue
            full_path = os.path.join(dirpath, fn)
            idx.setdefault(fn, []).append(full_path)
    return idx

def split_front_matter(text: str) -> Tuple[Optional[str], str, str]:
    newline = "\r\n" if "\r\n" in text else "\n"
    if not text.startswith("---"):
        return None, text, newline

    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return None, text, newline

    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return None, text, newline

    fm = "".join(lines[1:end_idx])
    body = "".join(lines[end_idx + 1 :])
    return fm, body, newline

def remove_top_level_keys(fm_lines: List[str], keys: set) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(fm_lines):
        line = fm_lines[i]
        if (not line.startswith((" ", "\t"))):
            m = re.match(r"^([A-Za-z0-9_\-]+)\s*:", line)
            if m and m.group(1) in keys:
                i += 1
                while i < len(fm_lines) and (not is_top_level_key_line(fm_lines[i])):
                    i += 1
                if out and out[-1].strip() == "" and i < len(fm_lines) and fm_lines[i].strip() == "":
                    i += 1
                continue
        out.append(line)
        i += 1
    while out and out[-1].strip() == "":
        out.pop()
    return out

def is_top_level_key_line(line: str) -> bool:
    if line.startswith((" ", "\t")):
        return False
    m = re.match(r"^([A-Za-z0-9_\-]+)\s*:", line)
    return bool(m)

def build_taxonomy_block(categories: List[str], tags: List[str]) -> List[str]:
    block: List[str] = []
    if categories:
        block.append("categories:")
        if len(categories) == 1:
            block.append(f"  - {yaml_scalar(categories[0])}")
        else:
            inner = ", ".join(yaml_scalar(c) for c in categories)
            block.append(f"  - [{inner}]")
    if tags:
        block.append("tags:")
        for t in tags:
            block.append(f"  - {yaml_scalar(t)}")
    return block

def normalize_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    return [str(x)]

def process_file(
    current_path: str, 
    root_dir: str,
    filename: str,
    categories: List[str], 
    tags: List[str], 
    dry_run: bool, 
    backup: bool
) -> str:
    # 1. 读取原内容
    try:
        with open(current_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        return f"[ERROR] Read failed: {e}"

    # 2. 处理 Front-matter
    fm, body, nl = split_front_matter(text)
    if fm is None:
        return "[SKIP] No Front-Matter found"

    fm_lines = fm.splitlines()
    fm_lines = remove_top_level_keys(fm_lines, {"tags", "tag", "categories", "category"})
    block = build_taxonomy_block(categories, tags)
    
    if block:
        if fm_lines and fm_lines[-1].strip() != "":
            fm_lines.append("")
        fm_lines.extend(block)

    new_fm = nl.join(fm_lines).rstrip() + nl
    new_content = f"---{nl}{new_fm}---{nl}{body}"

    # 3. 计算目标路径 (Markdown)
    if categories:
        sub_dir = to_snake_case(categories[0])
        target_dir = os.path.join(root_dir, sub_dir)
    else:
        target_dir = root_dir

    target_path = os.path.join(target_dir, filename)
    
    abs_current = os.path.abspath(current_path)
    abs_target = os.path.abspath(target_path)
    path_changed = (abs_current != abs_target)
    content_changed = (new_content != text)

    # 4. 计算 Asset Folder (资源目录) 路径
    # Hexo 资源目录名通常是不带后缀的文件名
    filename_no_ext = os.path.splitext(filename)[0]
    current_asset_dir = os.path.join(os.path.dirname(current_path), filename_no_ext)
    target_asset_dir = os.path.join(target_dir, filename_no_ext)
    
    has_asset_dir = os.path.isdir(current_asset_dir)
    
    # 5. 冲突检查
    if not path_changed and not content_changed:
        return "[NO CHANGE]"

    if path_changed:
        if os.path.exists(abs_target):
            return f"[SKIP] Target MD exists: {sub_dir}/{filename}"
        if has_asset_dir and os.path.exists(target_asset_dir):
             return f"[SKIP] Target Asset Dir exists: {sub_dir}/{filename_no_ext}/"

    # 6. 构造操作日志
    action_msg = []
    if content_changed: action_msg.append("UPDATE_FM")
    if path_changed: 
        msg = f"MOVE -> {sub_dir}/"
        if has_asset_dir:
            msg += " (+ASSETS)"
        action_msg.append(msg)
    
    final_msg = " + ".join(action_msg)

    if dry_run:
        return f"[DRY-RUN] {final_msg}"

    # 7. 执行写入与移动
    try:
        if path_changed:
            os.makedirs(target_dir, exist_ok=True)
        
        # 写入新 MD 文件
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        
        # 备份 (仅备份 Markdown)
        if backup:
            with open(target_path + ".bak", "w", encoding="utf-8") as bf:
                bf.write(text)

        # 移动资源目录
        if path_changed and has_asset_dir:
            # shutil.move 会把目录整体移过去
            shutil.move(current_asset_dir, target_asset_dir)

        # 删除旧 MD 文件
        if path_changed and os.path.exists(abs_current):
            os.remove(abs_current)

        return f"[OK] {final_msg}"

    except Exception as e:
        # 简单的回滚逻辑很难做完美，建议操作前 git commit
        return f"[ERROR] Write/Move failed: {e}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, help="你的 mapping json 文件路径")
    ap.add_argument("--root", required=True, help="Hexo markdown 根目录 (_posts)")
    ap.add_argument("--dry-run", action="store_true", help="只输出操作，不修改磁盘")
    ap.add_argument("--no-backup", action="store_true", help="不生成 .bak 备份")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"Error: root directory '{args.root}' does not exist.")
        sys.exit(1)

    posts = load_mapping(args.map)
    idx = index_markdown_files(args.root)

    print(f"Found {len(idx)} markdown files in {args.root}")
    
    processed_count = 0
    missing_count = 0
    ambiguous_count = 0

    for p in posts:
        fn = p.get("file")
        if not fn:
            continue

        paths = idx.get(fn, [])
        if not paths:
            missing_count += 1
            continue
        if len(paths) > 1:
            ambiguous_count += 1
            print(f"[AMBIGUOUS] {fn} found in multiple places: {paths}")
            continue

        categories = normalize_list(p.get("categories"))
        tags = normalize_list(p.get("tags"))
        
        res = process_file(
            current_path=paths[0],
            root_dir=args.root,
            filename=fn,
            categories=categories,
            tags=tags,
            dry_run=args.dry_run,
            backup=(not args.no_backup)
        )
        
        if not res.startswith("[NO CHANGE]"):
            print(f"{fn}: {res}")
            processed_count += 1

    print(f"\nDone. Processed: {processed_count}, Missing: {missing_count}, Ambiguous: {ambiguous_count}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)


# python3 source/_posts/butterfly/sync_hexo_taxonomy.py --map source/_posts/butterfly/post_tags.json --root ./source/_posts --dry-run 
# python3 source/_posts/butterfly/sync_hexo_taxonomy.py --map source/_posts/butterfly/post_tags.json --root ./source/_posts --no-backup
