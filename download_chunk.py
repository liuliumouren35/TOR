#!/usr/bin/env python3
"""
通过 HuggingFace Hub API 直接下载 MedRAG/textbooks 的 chunk 文件，绕过 Git LFS。
当 git lfs pull 因网络/镜像问题失败时使用此脚本。

在项目根目录 TOR 下执行: python download_chunk.py
"""
import os
import time

# 与 metadatas.jsonl 中 source 一致的 18 个 chunk 文件名（无扩展名）
CHUNK_NAMES = [
    "Anatomy_Gray", "Biochemistry_Lippincott", "Cell_Biology_Alberts",
    "First_Aid_Step1", "First_Aid_Step2", "Gynecology_Novak", "Histology_Ross",
    "Immunology_Janeway", "InternalMed_Harrison", "Neurology_Adams",
    "Obstentrics_Williams", "Pathology_Robbins", "Pathoma_Husain",
    "Pediatrics_Nelson", "Pharmacology_Katzung", "Physiology_Levy",
    "Psichiatry_DSM-5", "Surgery_Schwartz",
]


def main():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("请先安装: pip install huggingface_hub")
        return 1

    root = os.path.dirname(os.path.abspath(__file__))
    corpus_root = os.path.join(root, "corpus", "textbooks")
    chunk_dir = os.path.join(corpus_root, "chunk")
    os.makedirs(chunk_dir, exist_ok=True)

    # 使用项目内缓存，避免 ~/.cache 权限问题
    cache_dir = os.path.join(root, "corpus", ".cache_hf")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = cache_dir

    print("正在通过 HuggingFace Hub 下载 MedRAG/textbooks 的 chunk/*.jsonl ...")
    print("目标目录:", chunk_dir)
    print("缓存目录:", cache_dir)

    done = 0
    for name in CHUNK_NAMES:
        filename = f"chunk/{name}.jsonl"
        dest = os.path.join(chunk_dir, f"{name}.jsonl")
        if os.path.exists(dest) and os.path.getsize(dest) > 100:
            print(f"  跳过（已存在）: {name}.jsonl")
            done += 1
            continue
        for attempt in range(3):
            try:
                path = hf_hub_download(
                    repo_id="MedRAG/textbooks",
                    filename=filename,
                    repo_type="dataset",
                    local_dir=corpus_root,
                    local_dir_use_symlinks=False,
                    cache_dir=cache_dir,
                    resume_download=True,
                )
                print(f"  已下载: {name}.jsonl")
                done += 1
                break
            except Exception as e:
                print(f"  {name}.jsonl 第 {attempt + 1} 次失败: {e}")
                if attempt < 2:
                    time.sleep(5)
                else:
                    print(f"  跳过: {name}.jsonl")
        time.sleep(0.5)

    n = len([f for f in os.listdir(chunk_dir) if f.endswith(".jsonl") and os.path.getsize(os.path.join(chunk_dir, f)) > 100])
    print("完成。chunk/ 下有效 .jsonl 文件数:", n)
    return 0 if n >= 18 else 1


if __name__ == "__main__":
    exit(main())
