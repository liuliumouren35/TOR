#!/usr/bin/env python3
"""
下载支撑 RAG 的语料：目前仅从 HuggingFace 拉取 Textbooks 的 chunk。
StatPearls 在 HF 上无现成 chunk，需按下方说明从 NCBI 下载并本地生成。

在项目根目录 TOR 下执行: python download_all_corpora.py
依赖: pip install huggingface_hub

StatPearls 获取方式（需手动执行，在项目根目录下）：
  git clone https://hf-mirror.com/datasets/MedRAG/statpearls corpus/statpearls  # 若尚未 clone
  wget https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz -P corpus/statpearls
  tar -xzvf corpus/statpearls/statpearls_NBK430685.tar.gz -C corpus/statpearls
  python src/data/statpearls.py
"""
import os
import time

# Textbooks: 与 metadatas 一致的 18 个 chunk 名（MedRAG/textbooks）
TEXTBOOKS_CHUNKS = [
    "Anatomy_Gray", "Biochemistry_Lippincott", "Cell_Biology_Alberts",
    "First_Aid_Step1", "First_Aid_Step2", "Gynecology_Novak", "Histology_Ross",
    "Immunology_Janeway", "InternalMed_Harrison", "Neurology_Adams",
    "Obstentrics_Williams", "Pathology_Robbins", "Pathoma_Husain",
    "Pediatrics_Nelson", "Pharmacology_Katzung", "Physiology_Levy",
    "Psichiatry_DSM-5", "Surgery_Schwartz",
]

# 仅 Textbooks 在 HF 上有现成 chunk；StatPearls 需从 NCBI 下载后用 src/data/statpearls.py 生成
CORPORA = [
    ("MedRAG/textbooks", "textbooks", TEXTBOOKS_CHUNKS),
]


def download_repo_chunks(repo_id, corpus_name, chunk_names, root, cache_dir):
    """下载一个 repo 的 chunk/*.jsonl 到 corpus/{corpus_name}/chunk/。"""
    from huggingface_hub import hf_hub_download, list_repo_files

    corpus_root = os.path.join(root, "corpus", corpus_name)
    chunk_dir = os.path.join(corpus_root, "chunk")
    os.makedirs(chunk_dir, exist_ok=True)

    if chunk_names is not None:
        # 已知文件名列表（如 textbooks）
        files_to_download = [f"chunk/{n}.jsonl" for n in chunk_names]
    else:
        # 从 repo 列出 chunk/*.jsonl
        try:
            all_files = list_repo_files(repo_id, repo_type="dataset")
            files_to_download = [f for f in all_files if f.startswith("chunk/") and f.endswith(".jsonl")]
        except Exception as e:
            print(f"  列出文件失败: {e}")
            return 0
        if not files_to_download:
            print(f"  未找到 chunk/*.jsonl")
            return 0

    print(f"  共 {len(files_to_download)} 个 chunk 文件")
    done = 0
    for i, filename in enumerate(files_to_download):
        name = os.path.basename(filename)
        dest = os.path.join(chunk_dir, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 100:
            done += 1
            if (i + 1) % 50 == 0 or i == 0 or i == len(files_to_download) - 1:
                print(f"  进度: {i+1}/{len(files_to_download)} (已跳过/完成 {done})")
            continue
        for attempt in range(3):
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=corpus_root,
                    local_dir_use_symlinks=False,
                    cache_dir=cache_dir,
                    resume_download=True,
                )
                done += 1
                if (i + 1) % 20 == 0 or i == len(files_to_download) - 1:
                    print(f"  进度: {i+1}/{len(files_to_download)}")
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(5)
                else:
                    print(f"  跳过 {name}: {e}")
        time.sleep(0.3)

    n = len([f for f in os.listdir(chunk_dir) if f.endswith(".jsonl") and os.path.getsize(os.path.join(chunk_dir, f)) > 100])
    return n


def main():
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("请先安装: pip install huggingface_hub")
        return 1

    root = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(root, "corpus", ".cache_hf")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = cache_dir
    print("缓存目录:", cache_dir)
    print()

    total_ok = 0
    for repo_id, corpus_name, chunk_list in CORPORA:
        print("=" * 60)
        print(f"语料: {corpus_name} ({repo_id})")
        try:
            n = download_repo_chunks(repo_id, corpus_name, chunk_list, root, cache_dir)
            print(f"  完成: {corpus_name}/chunk/ 下有效 .jsonl 数 = {n}")
            total_ok += 1
        except Exception as e:
            print(f"  失败: {e}")
        print()

    print("=" * 60)
    print(f"完成。已下载语料: corpus/textbooks")
    print()
    print("若需 StatPearls（主诉科室用）：HF 上无现成 chunk，请在本项目根目录执行：")
    print("  wget https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz -P corpus/statpearls")
    print("  tar -xzvf corpus/statpearls/statpearls_NBK430685.tar.gz -C corpus/statpearls")
    print("  python src/data/statpearls.py")
    return 0 if total_ok == len(CORPORA) else 1


if __name__ == "__main__":
    exit(main())
