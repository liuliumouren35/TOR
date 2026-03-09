#!/usr/bin/env python3
"""
测试 RAG 检索：用 MedCPT + Textbooks 语料做一次检索，并可选地调用 LLM 生成回答。

用法（在项目根目录 TOR 下执行）：
  python test_rag.py                    # 只测试检索，打印检索到的文档
  python test_rag.py --llm               # 检索 + 用 DeepSeek 生成一段回答（需配置 API Key）

首次运行会自动下载语料和索引（约数 GB），需要网络和一定时间。
"""
import sys
import os
import argparse

# 保证从项目根目录运行时能导入 src 下的模块
if __name__ == "__main__":
    _root = os.path.dirname(os.path.abspath(__file__))
    _src = os.path.join(_root, "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

# 默认在项目根目录下放 corpus
CORPUS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")


def test_retrieve_only(question: str, k: int = 3, corpus_dir: str = None):
    """只测试检索：不调用 LLM，只打印检索到的文档片段。"""
    from utils import RetrievalSystem

    db_dir = corpus_dir or CORPUS_DIR
    print("正在初始化检索系统（MedCPT + Textbooks），首次运行会下载语料与索引...")
    retrieval = RetrievalSystem(
        retriever_name="MedCPT",
        corpus_name="Textbooks",
        db_dir=db_dir,
    )
    print("开始检索...")
    snippets, scores = retrieval.retrieve(question, k=k)
    print("\n" + "=" * 60)
    print("检索结果（前 {} 条）".format(len(snippets)))
    print("=" * 60)
    for i, (snip, sc) in enumerate(zip(snippets, scores), 1):
        title = snip.get("title", "(无标题)")
        content = (snip.get("content") or "")[:500]
        if len((snip.get("content") or "")) > 500:
            content += "..."
        print("\n[{}] 相关度: {:.4f}".format(i, sc))
        print("标题:", title)
        print("内容预览:", content)
    return snippets, scores


def test_rag_with_llm(question: str, k: int = 3, corpus_dir: str = None, min_score: float = None):
    """检索 + LLM：先检索，再用 DeepSeek 根据检索结果生成回答。min_score 为 None 时不按分数过滤。"""
    from utils import RetrievalSystem

    db_dir = corpus_dir or CORPUS_DIR
    print("正在初始化检索系统（MedCPT + Textbooks）...")
    retrieval = RetrievalSystem(
        retriever_name="MedCPT",
        corpus_name="Textbooks",
        db_dir=db_dir,
    )
    print("检索中...")
    snippets, scores = retrieval.retrieve(question, k=k)
    # 可选：只保留“相关度足够”的文档再交给 LLM（分数为模型内积，阈值需按当前 MedCPT 经验设定）
    if min_score is not None:
        kept = [(s, sc) for s, sc in zip(snippets, scores) if sc >= min_score]
        if not kept:
            print("没有文档达到最小相关度 {:.2f}，跳过 LLM 调用。".format(min_score))
            return
        snippets, scores = [x[0] for x in kept], [x[1] for x in kept]
        print("按最小相关度 {:.2f} 过滤后保留 {} 条文档。".format(min_score, len(snippets)))
    retrieved_text = "\n\n".join(
        'Document [{}] (Title: {}) {}'.format(
            idx, s.get("title", ""), (s.get("content") or "")[:800]
        )
        for idx, s in enumerate(snippets)
    )
    print("\n检索到的文档已作为上下文，正在调用 LLM...")
    try:
        from openai import OpenAI
        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("未设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY，跳过 LLM 调用。")
            print("检索结果预览：\n", retrieved_text[:1500], "...")
            return
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一名医学助手。请根据给定的文档片段简要回答用户问题。"},
                {"role": "user", "content": "参考文档：\n\n{}\n\n用户问题：{}".format(retrieved_text, question)},
            ],
            stream=False,
        )
        answer = resp.choices[0].message.content
        print("\n" + "=" * 60)
        print("RAG 回答")
        print("=" * 60)
        print(answer)
    except Exception as e:
        print("LLM 调用失败:", e)


def main():
    parser = argparse.ArgumentParser(description="测试 TOR 项目的 RAG 检索（及可选 LLM）")
    parser.add_argument("--question", "-q", default="What are the symptoms of type 2 diabetes?",
                        help="检索问题（英文医学问题效果更好）")
    parser.add_argument("--k", type=int, default=3, help="检索返回的文档数量")
    parser.add_argument("--llm", action="store_true", help="检索后调用 DeepSeek 生成回答（需设置 DEEPSEEK_API_KEY）")
    parser.add_argument("--corpus-dir", default=None, help="语料目录，默认为项目下的 corpus/")
    parser.add_argument("--min-score", type=float, default=None, dest="min_score",
                        help="仅 --llm 时有效：只把相关度 >= 该值的文档交给 LLM（MedCPT 内积，经验值约 50–60；不设则不过滤）")
    args = parser.parse_args()

    if args.llm:
        test_rag_with_llm(args.question, k=args.k, corpus_dir=args.corpus_dir, min_score=args.min_score)
    else:
        test_retrieve_only(args.question, k=args.k, corpus_dir=args.corpus_dir)


if __name__ == "__main__":
    main()
