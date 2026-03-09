# TOR

Repo for **Tree-of-Reasoning: Towards Complex Medical Diagnosis via Multi-Agent Reasoning with Evidence Tree**.

## 快速测试 RAG

只测检索或「检索 + LLM」完整流程，见 **[RAG_TEST.md](RAG_TEST.md)**。  
在项目根目录执行：

```bash
pip install -r requirements.txt
python test_rag.py              # 只测检索
python test_rag.py --llm         # 检索 + DeepSeek 回答（需设置 DEEPSEEK_API_KEY）
```

## 使用 CMB-Clin 跑多智能体诊断

本仓库的病例输入为「主诉 / 现病史 / 体格检查 / 实验室 / 影像 / 病理 + 多选题 options/label」的 JSON。  
若使用 **CMB-Clin**（74 例复杂病例），可用脚本整理成上述格式：

```bash
pip install datasets   # 若从 HuggingFace 拉取
python scripts/cmb_clin_to_tor.py --output data/tor_cases
```

再将 `src/discuss_merge_3.py` 中的 `case_dir` 改为 `data/tor_cases` 即可。  
说明：CMB-Clin 为开放式问答，脚本会从首条 QA 的答案中抽取「诊断」并生成单选项（A: 正确诊断），保证流程可跑；若要严格评估多选准确率，可自行为 74 例补充干扰项（如从 CMB-Exam 按科室选题）。
