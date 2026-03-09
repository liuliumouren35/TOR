#!/usr/bin/env python3
"""
将 CMB-Clin 数据整理为 TOR 实验所需的 JSON 病例格式。

用法:
  # 从 HuggingFace 拉取 CMB-Clin 并转换（需 pip install datasets）
  python scripts/cmb_clin_to_tor.py --output data/tor_cases

  # 从本地 JSON 文件转换（CMB 仓库 data 目录下的 clin 数据）
  python scripts/cmb_clin_to_tor.py --input path/to/cmb_clin.json --output data/tor_cases

输出: 每个病例一个 JSON 文件，可直接被 discuss_merge_3 的 case_dir 使用。
"""

import argparse
import json
import os
import re
import sys


# ---------- 从 CMB-Clin description 文本中解析出 TOR 所需字段 ----------

def _extract_age_sex(text: str) -> tuple[str, str]:
    """从 病史摘要 或 现病史 中解析年龄、性别。如：病人，男，49岁 / 患者，女，38岁"""
    age, sex = "", ""
    # 男/女
    m = re.search(r"[病人患者].*?[，,]\s*([男女]).*?[，,]\s*(\d+)\s*岁", text)
    if m:
        sex = "男" if m.group(1) == "男" else "女"
        age = m.group(2)
    if not age and re.search(r"(\d+)\s*岁", text):
        age = re.search(r"(\d+)\s*岁", text).group(1)
    if not sex and re.search(r"[，,]\s*([男女])\s*[，,]?", text):
        sex = re.search(r"[，,]\s*([男女])\s*[，,]?", text).group(1)
    return age or "", sex or ""


def _section_after(s: str, marker: str, next_markers: list[str] | None = None) -> str:
    """在 s 中找 marker 后的内容，直到下一个 next_markers 或结尾。"""
    if marker not in s:
        return ""
    start = s.index(marker)
    # 从 marker 后第一个换行之后开始取内容
    rest = s[start + len(marker):].lstrip()
    if rest.startswith("\n"):
        rest = rest.lstrip()
    # 去掉开头的（1）（2）等
    rest = re.sub(r"^[（(]\d+[)）]\s*", "", rest)
    if next_markers:
        for nm in next_markers:
            if nm in rest:
                rest = rest[: rest.index(nm)].rstrip()
    return rest.strip()


def _split_sections(description: str) -> dict[str, str]:
    """按 现病史、体格检查、辅助检查 等大标题切分。返回各段纯文本（不含标题）。"""
    description = description.replace("\r\n", "\n")
    sections = {}
    # 现病史（到体格检查或辅助检查为止）
    for start_marker in ["现病史", "体格检查", "辅助检查"]:
        content = _section_after(
            description,
            start_marker,
            next_markers=["体格检查", "辅助检查", "现病史"],
        )
        if start_marker == "现病史" and not content and "病史摘要" in description:
            # 整块 现病史\n（1）病史摘要...（2）主诉...
            idx = description.find("现病史")
            end = len(description)
            for end_marker in ["体格检查", "辅助检查"]:
                if end_marker in description[idx:]:
                    end = idx + description[idx:].index(end_marker)
                    break
            content = description[idx + len("现病史"): end].strip()
        sections[start_marker] = content
    return sections


def _extract_chief_complaint(present_illness_block: str) -> str:
    """从现病史块中抽出 主诉 一行或一段。"""
    if not present_illness_block:
        return ""
    # （2）主诉\n     右下腹痛并自扪及包块3小时。
    m = re.search(r"[（(]2[)）]\s*主诉\s*[：:\s]*\n?\s*([^\n]+)", present_illness_block)
    if m:
        return (m.group(1) or "").strip()
    if "主诉" in present_illness_block:
        after = present_illness_block.split("主诉", 1)[-1].strip()
        first_line = after.split("\n")[0].strip()
        return re.sub(r"^[：:\s]+", "", first_line)
    return ""


def _extract_present_illness(present_illness_block: str) -> str:
    """现病史：病史摘要 + 病情发展，可保留整块或去掉主诉重复部分。"""
    if not present_illness_block:
        return ""
    # 去掉（1）（2）编号，保留内容
    text = re.sub(r"[（(]\d+[)）]\s*", "", present_illness_block)
    return text.strip()


def _parse_aux_exam(aux_block: str) -> dict[str, str]:
    """解析 辅助检查：实验室、超声、X线、CT、磁共振、病理 等。"""
    out = {
        "Laboratory-Examination": "",
        "X光影像检查": "",
        "CT影像检查": "",
        "磁共振影像检查": "",
        "超声影像检查": "",
        "病理检查": "",
    }
    if not aux_block:
        return out
    # 按（1）（2）（3）分段，再按关键词归类
    parts = re.split(r"[（(]\d+[)）]\s*", aux_block)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if any(k in part for k in ["实验室检查", "血常规", "尿常规", "生化"]):
            out["Laboratory-Examination"] += ("\n" + part) if out["Laboratory-Examination"] else part
        elif any(k in part for k in ["超声", "B超", "多普勒"]):
            out["超声影像检查"] += ("\n" + part) if out["超声影像检查"] else part
        elif "X线" in part or "X光" in part:
            out["X光影像检查"] += ("\n" + part) if out["X光影像检查"] else part
        elif "CT" in part:
            out["CT影像检查"] += ("\n" + part) if out["CT影像检查"] else part
        elif "磁共振" in part or "MRI" in part:
            out["磁共振影像检查"] += ("\n" + part) if out["磁共振影像检查"] else part
        elif "病理" in part:
            out["病理检查"] += ("\n" + part) if out["病理检查"] else part
        else:
            # 未明确归类则归实验室
            out["Laboratory-Examination"] += ("\n" + part) if out["Laboratory-Examination"] else part
    return out


def _extract_diagnosis_from_solution(solution: str) -> str:
    """从 QA 的 solution 中提取 诊断：xxx。"""
    if not solution:
        return ""
    for line in solution.split("\n"):
        line = line.strip()
        if line.startswith("诊断：") or line.startswith("诊断:"):
            return line.replace("诊断：", "").replace("诊断:", "").strip()
    if "诊断" in solution:
        m = re.search(r"诊断[：:\s]+([^\n]+)", solution)
        if m:
            return m.group(1).strip()
    return ""


def parse_cmb_clin_item(item: dict) -> dict | None:
    """
    将 CMB-Clin 的一条 item 转为 TOR 所需的一条 case（单条 JSON 的根结构）。
    若缺少 description 或 QA_pairs 则返回 None。
    """
    description = item.get("description") or item.get("Description") or ""
    qa_pairs = item.get("QA_pairs", [])
    if not description.strip():
        return None

    sections = _split_sections(description)
    present_block = sections.get("现病史", "")
    chief = _extract_chief_complaint(present_block)
    present_illness = _extract_present_illness(present_block)
    physical = sections.get("体格检查", "")
    aux = _parse_aux_exam(sections.get("辅助检查", ""))

    age, sex = _extract_age_sex(present_block or description)

    diagnosis = ""
    if qa_pairs and len(qa_pairs) > 0:
        first_solution = qa_pairs[0].get("solution") or qa_pairs[0].get("Solution") or ""
        diagnosis = _extract_diagnosis_from_solution(first_solution)

    # TOR 需要 options 与 label。CMB-Clin 只有开放式问答，没有选项；
    # 这里用「A: 正确诊断」单选项，保证流程可跑；评估时可按需改为多选项或自由文本比对。
    options_str = f"A: {diagnosis}" if diagnosis else "A: 见病例分析"
    label = "A"

    return {
        "Age": age,
        "Sex": sex,
        "Chief-Complaints": chief,
        "Present-Illness": present_illness,
        "Physical-Examination": physical,
        "Laboratory-Examination": aux.get("Laboratory-Examination", ""),
        "X光影像检查": aux.get("X光影像检查", ""),
        "CT影像检查": aux.get("CT影像检查", ""),
        "磁共振影像检查": aux.get("磁共振影像检查", ""),
        "超声影像检查": aux.get("超声影像检查", ""),
        "病理检查": aux.get("病理检查", ""),
        "Diagnosis": diagnosis,
        "options": options_str,
        "label": label,
    }


def load_cmb_clin_from_hf():
    """从 HuggingFace 加载 CMB-Clin。需要 datasets。"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("从 HF 加载需安装: pip install datasets")
    ds = load_dataset("FreedomIntelligence/CMB", "clin")
    # 取 train 或 first split
    if "train" in ds:
        return list(ds["train"])
    return list(ds[list(ds.keys())[0]])


def load_cmb_clin_from_json(path: str) -> list[dict]:
    """从本地 JSON 加载。支持单条对象或 list。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def main():
    parser = argparse.ArgumentParser(description="CMB-Clin -> TOR 病例 JSON")
    parser.add_argument("--input", "-i", type=str, default="", help="本地 CMB-Clin JSON 路径（不填则从 HF 拉取）")
    parser.add_argument("--output", "-o", type=str, default="data/tor_cases", help="输出目录，每病例一个 JSON")
    parser.add_argument("--prefix", type=str, default="case", help="输出文件名前缀，如 case -> case_000.json")
    args = parser.parse_args()

    if args.input and os.path.isfile(args.input):
        items = load_cmb_clin_from_json(args.input)
        print(f"从本地加载 {len(items)} 条: {args.input}")
    else:
        if args.input:
            print(f"未找到文件 {args.input}，改为从 HuggingFace 拉取 CMB-Clin")
        items = load_cmb_clin_from_hf()
        print(f"从 HuggingFace 加载 {len(items)} 条 CMB-Clin")

    os.makedirs(args.output, exist_ok=True)
    written = 0
    for i, item in enumerate(items):
        case = parse_cmb_clin_item(item)
        if case is None:
            continue
        out_path = os.path.join(args.output, f"{args.prefix}_{i:03d}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(case, f, ensure_ascii=False, indent=2)
        written += 1
    print(f"已写入 {written} 个病例到 {args.output}，可直接将 discuss_merge_3 的 case_dir 设为该目录。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
