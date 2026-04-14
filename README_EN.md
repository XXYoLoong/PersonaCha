# PersonaCha

[简体中文说明](./README.md)

> An open-source practice and research repository for **persona-grounded dialogue**, focused on classic dataset reproduction, lightweight model training, experiment tracking, and future extensions.

## Overview

PersonaCha is designed as a structured entry point for reproducing persona-grounded dialogue generation experiments. The project centers on **PersonaChat** and **Synthetic-Persona-Chat**, with **BlenderBot Small** and **DialoGPT-small** as recommended starter baselines.

Rather than building a full production chatbot, this repository aims to provide a clear and extensible research path: understand the task from public datasets, train lightweight baselines, and analyze persona consistency, dialogue quality, and the impact of synthetic data augmentation.

## Goals

- Reproduce the standard training pipeline for persona-grounded dialogue
- Compare explicit persona data with synthetic persona data augmentation
- Build lightweight baselines suitable for coursework and early-stage research
- Prepare for future extensions such as Chinese persona datasets and long-horizon consistency studies

## Task Definition

The core objective of persona-grounded dialogue generation is to produce responses that are both contextually appropriate and consistent with a given persona profile across multi-turn conversations.

Typical research questions include:

- How should persona information be injected into the model?
- Can the model maintain persona consistency over multiple turns?
- Does synthetic persona dialogue improve response quality or persona adherence?
- How should automatic and human evaluation be combined?

## Recommended Datasets

### 1. PersonaChat

The classic benchmark dataset and the most common starting point for this task.

- Paper: <https://huggingface.co/papers/1801.07243>
- Dataset: <https://huggingface.co/datasets/bavard/personachat_truecased>

### 2. Synthetic-Persona-Chat

A high-quality persona dialogue dataset expanded with large language models, useful for data augmentation and controlled comparison.

- Paper: <https://huggingface.co/papers/2312.10007>
- Dataset: <https://huggingface.co/datasets/google/Synthetic-Persona-Chat>

### 3. RealPersonaChat

A more realistic persona dialogue resource grounded in speakers' own personalities.

- Dataset: <https://huggingface.co/datasets/nu-dialogue/real-persona-chat>

## Recommended Baseline Models

### BlenderBot Small 90M

A lightweight seq2seq baseline suitable for open-domain response generation.

- Model: <https://huggingface.co/facebook/blenderbot_small-90M>

### DialoGPT-small

A lightweight causal language modeling baseline for dialogue generation.

- Model: <https://huggingface.co/microsoft/DialoGPT-small>

## Suggested Experiment Path

### Experiment 1: PersonaChat Baseline Fine-tuning

Train BlenderBot Small or DialoGPT-small on PersonaChat to establish a lightweight baseline for persona-aware response generation.

### Experiment 2: Synthetic Data Augmentation

Introduce Synthetic-Persona-Chat as additional training data or a second-stage fine-tuning corpus, then compare response quality and persona consistency.

### Experiment 3: Error Analysis

Perform case-based inspection of generated responses, especially focusing on:

- persona conflicts
- repetitive responses
- dialogue context forgetting
- style drift

## Suggested Repository Structure

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

## Environment Setup

Python 3.10+ is recommended, and the Hugging Face ecosystem is the preferred stack for data processing and training.

```bash
git clone https://github.com/XXYoLoong/PersonaCha.git
cd PersonaCha
python -m venv .venv
source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
pip install -U transformers datasets accelerate sentencepiece evaluate
```

## Evaluation Suggestions

Automatic metrics alone are often insufficient for persona-grounded dialogue. A stronger evaluation protocol should consider:

- **Fluency**: whether the response is natural and grammatically sound
- **Coherence**: whether the response fits the dialogue context
- **Persona Consistency**: whether the response aligns with the persona profile
- **Diversity**: whether the model avoids overly templated outputs

Whenever possible, add human evaluation or qualitative case analysis for more reliable judgment.

## Use Cases

This repository is especially suitable for:

- coursework and lab assignments
- early-stage paper preparation
- lightweight model reproduction and fine-tuning practice
- validating English baselines before building Chinese persona datasets

## References

1. **Personalizing Dialogue Agents: I have a dog, do you have pets too?**  
   <https://huggingface.co/papers/1801.07243>
2. **Faithful Persona-based Conversational Dataset Generation with Large Language Models**  
   <https://huggingface.co/papers/2312.10007>
3. **Two Tales of Persona in LLMs: A Survey of Role-Playing and Personalization**  
   <https://huggingface.co/papers/2406.01171>

## Notes

This repository currently serves as a public entry point for research and reproduction. A good workflow is to begin with a small and reproducible baseline, then extend toward Chinese persona dialogue, long-horizon consistency, and multi-agent data construction.

Issues and pull requests are welcome.