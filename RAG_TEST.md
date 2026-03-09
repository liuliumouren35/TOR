# 如何测试 RAG

按下面步骤可以在本项目中**只测检索**（不调 LLM），或**检索 + DeepSeek 生成回答**。

---

## 1. 环境准备

在项目根目录 `TOR` 下：

```bash
# 创建虚拟环境（推荐）
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

说明：若使用 GPU，可把 `faiss-cpu` 换成 `faiss-gpu`。

---

## 2. 只测检索（不调用 LLM）

在项目根目录执行：

```bash
python test_rag.py
```

或指定问题和返回条数：

```bash
python test_rag.py --question "What are the symptoms of hypertension?" --k 5
```

- **首次运行**会自动从 HuggingFace 拉取 **Textbooks** 语料，并从论文提供的链接下载 **MedCPT** 的预计算 embedding（约数 GB），需要较长时间和稳定网络。
- 运行成功后会在终端打印检索到的文档片段（标题 + 内容预览）。

---

## 3. 检索 + LLM 生成回答（完整 RAG）

先设置 DeepSeek API Key，再带 `--llm` 运行：

```bash
export DEEPSEEK_API_KEY="sk-f05f3d4564f44a7fb932264cf63d6b6f"
python test_rag.py --llm
```

或使用 OpenAI 的 Key（脚本会请求 DeepSeek 的 base_url，需确认 DeepSeek 是否支持）：

```bash
export OPENAI_API_KEY="你的 Key"
python test_rag.py --llm
```

脚本会先做检索，再把检索结果作为上下文发给 DeepSeek 生成一段回答。

---

## 4. 常见问题

| 情况 | 处理 |
|------|------|
| 报错 `No module named 'xxx'` | 在虚拟环境中执行 `pip install -r requirements.txt`，确保在 TOR 根目录运行 `python test_rag.py`。 |
| 首次运行卡在 “Downloading embeddings” | 正常。预计算索引较大，需等待下载完成；若 SharePoint 链接失效，可考虑用作者更新的链接或自行建索引。 |
| 想用其他语料（如 StatPearls） | 需改 `test_rag.py` 里 `RetrievalSystem(..., corpus_name="StatPearls", ...)`；StatPearls 会额外从 NCBI 下载并做 chunk，步骤见 `utils.py`。 |
| 语料想放到别的目录 | 使用 `--corpus-dir`：`python test_rag.py --corpus-dir /path/to/corpus`。 |
| `chunk/` 为空、报错 `FileNotFoundError: ... chunk/xxx.jsonl` 或 Git LFS 拉取失败 | 用脚本绕过 LFS：**只下 Textbooks** 执行 `python download_chunk.py` 或 `python download_all_corpora.py`（需 `pip install huggingface_hub`）。**StatPearls** 在 HF 上无现成 chunk，需从 NCBI 下 tar.gz 后执行 `python src/data/statpearls.py`，见脚本结束时的提示。 |
| 只想在“相关度够高”时才把文档交给 LLM | 使用 `--min-score`：`python test_rag.py --llm --min-score 55`（分数为 MedCPT 内积，经验范围约 50–65，可按需要调整；不设则不过滤）。 |

---

## 5. 脚本在做什么

- **只测检索**：用 `utils.RetrievalSystem`（MedCPT + Textbooks）对一个问题做检索，打印前 `k` 条文档。
- **加 `--llm`**：在检索基础上，把检索结果拼成上下文，调用 DeepSeek 生成回答；不依赖项目里的 `medrag` 或 `config`，只需环境变量中的 API Key。

更多细节见项目根目录下的 `test_rag.py` 顶部注释。
