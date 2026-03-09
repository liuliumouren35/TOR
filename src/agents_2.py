"""
多智能体医学诊断模块 (agents_2.py)
-----------------------------------
定义四类专科医生智能体：主诉医生、检验医生、影像医生、病理医生。
每个智能体继承 BaseDoctor，使用 MedRAG 做检索增强，并调用 LLM 生成证据树形式的诊断意见。
"""
import datetime
import json
import sys
from openai import OpenAI
from medrag import MedRAG

# 调用 DeepSeek 使用的模型名
MODEL_NAME = "deepseek-chat"

# 全局 LLM 客户端（需在运行前配置 api_key）
client = OpenAI(api_key="", base_url="https://api.deepseek.com")


class BaseDoctor:
    """所有专科医生的基类，封装 MedRAG 检索与通用文本处理逻辑。"""

    def __init__(self, llm_name="OpenAI/gpt-3.5-turbo-16k", retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus"):
        """初始化 MedRAG：检索器、语料库、数据库路径等可由子类覆盖。"""
        self.medrag = MedRAG(
            llm_name=llm_name,
            rag=True,
            follow_up=False,
            retriever_name=retriever_name,
            corpus_name=corpus_name,
            db_dir=db_dir
        )

    def process_medical_text(self, input_text, k=3):
        """对输入医学文本做 RAG 检索，返回格式化后的检索片段字符串。"""
        retrieved_snippets, scores = self.medrag.medrag_retrieve(input_text, k=k)

        retrieved_info = "\n".join([
            "Document [{:d}] (Title: {:s}) {:s}".format(
                idx, snippet["title"], snippet["content"]
            ) for idx, snippet in enumerate(retrieved_snippets)
        ])

        return {
            'retrieved_info': retrieved_info,
        }


# ---------- 主诉医生智能体 ----------
class ChiefComplaintDoctor(BaseDoctor):
    """主诉医生：根据主诉、现病史、体格检查做问诊与初步鉴别，输出主诉相关证据树。"""

    def __init__(self):
        super().__init__(
            llm_name="OpenAI/gpt-3.5-turbo-16k",
            retriever_name="MedCPT",
            corpus_name="StatPearls",
            db_dir="./corpus"
        )

    def examine_patient(self, patient_info):
        """根据患者信息（主诉、现病史、体格检查等）进行问诊分析，返回证据树及检索信息。"""
        print("The chief physician is currently handling the matter...")

        # 主诉 + 现病史 作为检索与推理的输入
        input_text = patient_info["Chief-Complaints"] + " " + patient_info["Present-Illness"]

        result = self.process_medical_text(input_text)

        template = """
You are an experienced attending physician responsible for detailed inquiry of the patient's chief complaints, medical history, and initial physical examination.

Based on the retrieved medical knowledge:
{retrieved_info}

Patient information is as follows:
{{
"Age": {age},
"Sex": {sex},
"Chief-Complaints": {chief_complaints},
"Present-Illness": {present_illness},
"Physical-Examination": {physical_examination}
}}

Please analyze the patient's information and output evidence tree structure in following format:
1. Clinical Clues: Identify key clinical clues present in the patient's chief complaints, medical history, and initial physical examination.
2. Possible Diseases: List possible diseases that these clinical clues might point to.
3. Reasoning Process: For each possible disease, provide a brief explanation of your reasoning process.
4. Evidence: For each possible disease, summarize the supporting evidence from the chief complaints, medical history, and physical examination.

Output Format:       
Chief Complaints Clinical Reasoning Pathway
├── Disease 1
│   └── Analysis: Brief explanation of the reasoning process
│       ├── Evidence 1: Chief Complaints
│       ├── Evidence 2: Medical History
│       └── Evidence 3: Physical Examination
├── Disease 2
│   └── Analysis: Brief explanation of the reasoning process
│       ├── Evidence 1: Chief Complaints
│       ├── Evidence 2: Medical History
│       └── Evidence 3: Physical Examination
└── ...

Please ensure that the output strictly follows the above format and only includes the evidence tree structure. Avoid any additional text or explanations outside the tree structure.
"""

        system_message = template.format(
            age=patient_info["Age"],
            sex=patient_info["Sex"],
            chief_complaints=patient_info["Chief-Complaints"],
            present_illness=patient_info["Present-Illness"],
            physical_examination=patient_info["Physical-Examination"],
            retrieved_info=result['retrieved_info']
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": "Please ensure that the output strictly follows the above format and only includes the evidence tree structure. Avoid any additional text or explanations outside the tree structure. "}
            ],
            stream=False
        )

        return {
            'prompt': system_message,
            'response': response.choices[0].message.content,
            'retrieved_info': result['retrieved_info']
        }


def chief_complaint_agent(patient_info):
    """便捷入口：创建主诉医生并执行问诊，返回证据树等结果。"""
    doctor = ChiefComplaintDoctor()
    return doctor.examine_patient(patient_info)


# ---------- 检验医生智能体 ----------
class LabDoctor(BaseDoctor):
    """检验医生：根据实验室检查结果分析异常指标，输出检验相关证据树。"""

    def __init__(self):
        super().__init__(
            llm_name="OpenAI/gpt-3.5-turbo-16k",
            retriever_name="MedCPT",
            corpus_name="Textbooks",
            db_dir="./corpus"
        )

    def analyze_results(self, lab_results):
        """分析检验结果文本，返回证据树及检索信息。"""
        print("The laboratory doctor is currently handling it...")

        result = self.process_medical_text(lab_results)

        template = """
You are an experienced laboratory physician responsible for analyzing laboratory test results and providing diagnostic suggestions based on the results.

Based on the retrieved medical knowledge:
{retrieved_info}

The laboratory test results are as follows:
{lab_results}

Please analyze the laboratory test results and output evidence tree structure in following format:
1. Abnormal Indicators: Identify which indicators in the laboratory test results are abnormal.
2. Possible Diseases: List possible diseases that these abnormal indicators might point to.
3. Reasoning Process: For each possible disease, provide a brief explanation of your reasoning process.
4. Evidence: For each possible disease, summarize the supporting evidence from the laboratory test results.

Output Format:       
Laboratory Test Clinical Reasoning Pathway
├── Disease 1
│   └── Analysis: Brief explanation of the reasoning process
│       ├── Evidence 1: Abnormal Indicator 1
│       ├── Evidence 2: Abnormal Indicator 2
│       └── Evidence 3: Abnormal Indicator 3
├── Disease 2
│   └── Analysis: Brief explanation of the reasoning process
│       ├── Evidence 1: Abnormal Indicator 1
│       ├── Evidence 2: Abnormal Indicator 2
│       └── Evidence 3: Abnormal Indicator 3
└── ...

Please ensure that the output strictly follows the above format and only includes the evidence tree structure. Avoid any additional text or explanations outside the tree structure. 
"""

        system_message = template.format(
            lab_results=lab_results,
            retrieved_info=result['retrieved_info']
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": "Please carefully consider and answer the above questions."}
            ],
            stream=False
        )
        
        return {
            'prompt': system_message,
            'response': response.choices[0].message.content,
            'retrieved_info': result['retrieved_info']
        }


def lab_agent(lab_results):
    """便捷入口：创建检验医生并分析检验结果。"""
    doctor = LabDoctor()
    return doctor.analyze_results(lab_results)


# ---------- 影像医生智能体 ----------
class ImagingDoctor(BaseDoctor):
    """影像医生：根据影像检查描述（X 光/CT/MRI/超声等）分析异常征象，输出影像证据树。"""

    def __init__(self):
        super().__init__(
            llm_name="OpenAI/gpt-3.5-turbo-16k",
            retriever_name="MedCPT",
            corpus_name="Textbooks",
            db_dir="./corpus"
        )

    def analyze_images(self, imaging_results):
        """分析影像检查结果文本，返回证据树及检索信息。"""
        print("The imaging doctor is currently processing it...")

        result = self.process_medical_text(imaging_results)

        template = """
You are an experienced imaging physician responsible for analyzing imaging test results and providing diagnostic suggestions based on the results.

Based on the retrieved medical knowledge:
{retrieved_info}

The imaging test results are as follows:
{imaging_results}

Please analyze the imaging test results and output evidence tree structure in following format:
1. Abnormal Findings: Identify what abnormal findings are present in the imaging test results.
2. Possible Diseases: List possible diseases that these abnormal findings might point to.
3. Reasoning Process: For each possible disease, provide a brief explanation of your reasoning process.
4. Evidence: For each possible disease, summarize the supporting evidence from the imaging test results.

Output Format:       
Imaging Test Clinical Reasoning Pathway
├── Disease 1
│   └── Analysis: Brief explanation of the reasoning process
│       ├── Evidence 1: Abnormal Finding 1
│       ├── Evidence 2: Abnormal Finding 2
│       └── Evidence 3: Abnormal Finding 3
├── Disease 2
│   └── Analysis: Brief explanation of the reasoning process
│       ├── Evidence 1: Abnormal Finding 1
│       ├── Evidence 2: Abnormal Finding 2
│       └── Evidence 3: Abnormal Finding 3
└── ...
Please ensure that the output strictly follows the above format and only includes the evidence tree structure. Avoid any additional text or explanations outside the tree structure.     """

        system_message = template.format(
            imaging_results=imaging_results,
            retrieved_info=result['retrieved_info']
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": "Please carefully consider and answer the above questions."}
            ],
            stream=False
        )
        
        return {
            'prompt': system_message,
            'response': response.choices[0].message.content,
            'retrieved_info': result['retrieved_info']
        }

def imaging_agent(imaging_results):
    """便捷入口：创建影像医生并分析影像结果。"""
    doctor = ImagingDoctor()
    return doctor.analyze_images(imaging_results)


# ---------- 病理医生智能体 ----------
class PathologyDoctor(BaseDoctor):
    """病理医生：根据病理检查结果分析异常发现，输出病理证据树。"""

    def __init__(self):
        super().__init__(
            llm_name="OpenAI/gpt-3.5-turbo-16k",
            retriever_name="MedCPT",
            corpus_name="Textbooks",
            db_dir="./corpus"
        )

    def analyze_pathology(self, pathology_results):
        """分析病理检查结果文本，返回证据树及检索信息。"""
        print("The pathologist is currently handling it...")

        result = self.process_medical_text(pathology_results)

        template = """
You are an experienced pathology physician responsible for analyzing pathology test results and providing diagnostic suggestions based on the results.

Based on the retrieved medical knowledge:
{retrieved_info}

The pathology test results are as follows:
{pathology_results}

Please analyze the pathology test results and output evidence tree structure in following format:
1. Abnormal Findings: Identify what abnormal findings are present in the pathology test results.
2. Possible Diseases: List possible diseases that these abnormal findings might point to.
3. Reasoning Process: For each possible disease, provide a brief explanation of your reasoning process.
4. Evidence: For each possible disease, summarize the supporting evidence from the pathology test results.

Output Format:       
Pathology Test Clinical Reasoning Pathway
├── Disease 1
│   └── Analysis: Brief explanation of the reasoning process
│       ├── Evidence 1: Abnormal Finding 1
│       ├── Evidence 2: Abnormal Finding 2
│       └── Evidence 3: Abnormal Finding 3
├── Disease 2
│   └── Analysis: Brief explanation of the reasoning process
│       ├── Evidence 1: Abnormal Finding 1
│       ├── Evidence 2: Abnormal Finding 2
│       └── Evidence 3: Abnormal Finding 3
└── ...

Please ensure that the output strictly follows the above format and only includes the evidence tree structure. Avoid any additional text or explanations outside the tree structure.
        """

        system_message = template.format(
            pathology_results=pathology_results,
            retrieved_info=result['retrieved_info']
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": "Please carefully consider and answer the above questions."}
            ],
            stream=False
        )
        
        return {
            'prompt': system_message,
            'response': response.choices[0].message.content,
            'retrieved_info': result['retrieved_info']
        }

def pathology_agent(pathology_results):
    """便捷入口：创建病理医生并分析病理结果。"""
    doctor = PathologyDoctor()
    return doctor.analyze_pathology(pathology_results)


def load_json_data(file_path):
    """从 JSON 病例文件加载并标准化为统一结构：患者信息、各模态检查结果、选项与标签。"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return {
        "patient_info": {
            "Age": data.get("Age", ""),
            "Sex": data.get("Sex", ""),
            "Chief-Complaints": data.get("Chief-Complaints", ""),
            "Present-Illness": data.get("Present-Illness", ""),
            "Physical-Examination": data.get("Physical-Examination", "")
        },
        "lab_results": data.get("Laboratory-Examination", ""),
        "imaging_results": "\n".join([
            data.get("X光影像检查") or "",
            data.get("CT影像检查") or "",
            data.get("磁共振影像检查") or "",
            data.get("超声影像检查") or ""
        ]).strip(),
        "pathology_results": data.get("病理检查", ""),
        "options": data.get("options", ""),
        "ground_truth": {
            "diagnosis": data.get("Diagnosis", ""),
            "options": data.get("options", ""),
            "label": data.get("label", "")
        }
    }
