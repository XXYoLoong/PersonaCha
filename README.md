# PersonaCha

[English README](./README_EN.md)

> 一个面向 **人格化对话（persona-grounded dialogue）** 的开源练手与研究仓库，聚焦于经典数据集复现、轻量模型训练、实验记录与后续扩展。

## 项目简介

PersonaCha 用于系统性复现和整理人格化对话生成任务的基础实验流程。项目以 **PersonaChat** 与 **Synthetic-Persona-Chat** 为核心数据来源，以 **BlenderBot Small** 和 **DialoGPT-small** 为推荐入门基线，适合作为课程作业、论文预研、工程练手和后续扩展研究的起点。

本仓库的目标不是做一个“大而全”的聊天系统，而是构建一条清晰、可复现、可扩展的研究路径：从公开数据集理解任务定义，到完成小模型训练，再到分析 persona 一致性、对话自然度和数据增强效果。

## 研究目标

- 复现经典 persona-grounded dialogue 任务的基本训练流程
- 对比显式 persona 数据与合成 persona 数据的训练效果
- 建立适合课程实验与论文预研的轻量化基线
- 为后续中文人格化对话数据构建和长程一致性研究打基础

## 任务定义

人格化对话生成的核心目标是：在多轮对话中，让模型生成的回复既符合当前上下文，又尽可能保持与给定 persona 描述一致。

典型研究问题包括：

- persona 信息如何注入模型
- 模型在多轮对话中能否维持设定一致性
- 合成数据能否帮助提升对话自然度和 persona adherence
- 自动评测与人工评测如何结合

## 推荐数据集

### 1. PersonaChat

经典的人格化对话数据集，是该方向最常见的入门基线。

- Paper: <https://huggingface.co/papers/1801.07243>
- Dataset: <https://huggingface.co/datasets/bavard/personachat_truecased>

### 2. Synthetic-Persona-Chat

基于大语言模型扩展生成的高质量 persona 对话数据集，可用于数据增强与对比实验。

- Paper: <https://huggingface.co/papers/2312.10007>
- Dataset: <https://huggingface.co/datasets/google/Synthetic-Persona-Chat>

### 3. RealPersonaChat

更强调真实说话者人格特征的 persona 对话资源，适合后续进一步研究。

- Dataset: <https://huggingface.co/datasets/nu-dialogue/real-persona-chat>

## 推荐基线模型

### BlenderBot Small 90M

适合作为 seq2seq 风格的轻量对话生成基线。

- Model: <https://huggingface.co/facebook/blenderbot_small-90M>

### DialoGPT-small

适合作为 causal language modeling 风格的轻量对话生成基线。

- Model: <https://huggingface.co/microsoft/DialoGPT-small>

## 建议实验路线

### 实验 1：PersonaChat 基线微调

在 PersonaChat 上训练 BlenderBot Small 或 DialoGPT-small，观察模型在短多轮对话中的基本生成能力。

### 实验 2：Synthetic-Persona-Chat 数据增强

将合成 persona 对话数据引入训练或继续微调，对比模型在 persona 一致性和回复自然度上的变化。

### 实验 3：误差分析

对生成结果进行案例分析，重点观察：

- persona 冲突
- 重复回答
- 对话上下文遗忘
- 风格漂移

## 建议目录结构

```text
PersonaCha/
├── README.md
├── README_EN.md
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── scripts/
├── src/
│   ├── data/
│   ├── training/
│   └── evaluation/
├── experiments/
│   ├── logs/
│   └── outputs/
└── docs/
```

## 环境建议

推荐使用 Python 3.10 及以上版本，并优先采用 Hugging Face 生态完成数据处理与训练。

```bash
git clone https://github.com/XXYoLoong/PersonaCha.git
cd PersonaCha
python -m venv .venv
source .venv/bin/activate  # Windows 请改为 .venv\Scripts\activate
pip install -U transformers datasets accelerate sentencepiece evaluate
```

## 评测建议

在 persona-grounded dialogue 任务中，单纯依赖自动指标通常不够。建议同时结合以下维度：

- **Fluency**：语言是否自然流畅
- **Coherence**：回复是否贴合上下文
- **Persona Consistency**：回复是否与 persona 设定一致
- **Diversity**：生成是否过于模板化

当条件允许时，建议加入人工评测或案例分析，以更可靠地判断 persona 生成质量。

## 适用场景

本仓库尤其适合以下用途：

- 课程作业与实验报告
- 人格化对话方向论文预研
- 小模型训练与复现练手
- 中文 persona 数据构建前的英文基线验证

## 参考文献

1. **Personalizing Dialogue Agents: I have a dog, do you have pets too?**  
   <https://huggingface.co/papers/1801.07243>
2. **Faithful Persona-based Conversational Dataset Generation with Large Language Models**  
   <https://huggingface.co/papers/2312.10007>
3. **Two Tales of Persona in LLMs: A Survey of Role-Playing and Personalization**  
   <https://huggingface.co/papers/2406.01171>

## 说明

本仓库当前定位为公开研究与复现入口。若你正在进行论文预研，可以先从小规模可复现实验开始，再逐步扩展到中文人格化对话、长程一致性和多 Agent 数据构建等方向。

欢迎通过 Issue 或 Pull Request 交流改进思路。