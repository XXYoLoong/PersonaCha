#!/usr/bin/env python3
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / 'scripts' / 'train_persona_easy.py'
OUTPUT_ROOT = ROOT / 'experiments' / 'outputs'
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
TRAIN_PROCESS = None


def detect_hardware() -> str:
    lines = [f'Platform: {platform.platform()}', f'Python: {sys.version.split()[0]}']
    try:
        import torch
        lines.append(f'CUDA available: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            lines.append(f'GPU count: {torch.cuda.device_count()}')
            for i in range(torch.cuda.device_count()):
                lines.append(f'GPU {i}: {torch.cuda.get_device_name(i)}')
                props = torch.cuda.get_device_properties(i)
                lines.append(f'GPU {i} VRAM: {round(props.total_memory / 1024**3, 2)} GB')
        else:
            lines.append('GPU: not detected by torch')
    except Exception as e:
        lines.append(f'torch check failed: {e}')
    return '\n'.join(lines)


def get_preset(name: str):
    presets = {
        'RTX 4060 8GB 推荐': {
            'model_name': 'facebook/blenderbot_small-90M',
            'dataset_name': 'bavard/personachat_truecased',
            'extra_dataset_name': '',
            'max_train_samples': 5000,
            'max_eval_samples': 500,
            'num_train_epochs': 1,
            'per_device_train_batch_size': 4,
            'per_device_eval_batch_size': 4,
            'gradient_accumulation_steps': 4,
        },
        '快速冒烟测试': {
            'model_name': 'facebook/blenderbot_small-90M',
            'dataset_name': 'bavard/personachat_truecased',
            'extra_dataset_name': '',
            'max_train_samples': 1000,
            'max_eval_samples': 100,
            'num_train_epochs': 1,
            'per_device_train_batch_size': 2,
            'per_device_eval_batch_size': 2,
            'gradient_accumulation_steps': 4,
        },
        '增强实验（含 Synthetic）': {
            'model_name': 'facebook/blenderbot_small-90M',
            'dataset_name': 'bavard/personachat_truecased',
            'extra_dataset_name': 'google/Synthetic-Persona-Chat',
            'max_train_samples': 5000,
            'max_eval_samples': 500,
            'num_train_epochs': 1,
            'per_device_train_batch_size': 2,
            'per_device_eval_batch_size': 2,
            'gradient_accumulation_steps': 8,
        },
    }
    return presets[name]


def apply_preset(preset_name: str):
    p = get_preset(preset_name)
    return p['model_name'], p['dataset_name'], p['extra_dataset_name'], p['max_train_samples'], p['max_eval_samples'], p['num_train_epochs'], p['per_device_train_batch_size'], p['per_device_eval_batch_size'], p['gradient_accumulation_steps']


def build_command(model_name, dataset_name, extra_dataset_name, max_train_samples, max_eval_samples, num_train_epochs, per_device_train_batch_size, per_device_eval_batch_size, gradient_accumulation_steps, dry_run):
    output_dir = OUTPUT_ROOT / 'blenderbot_personachat'
    cmd = [
        sys.executable,
        str(SCRIPT),
        '--model_name', str(model_name),
        '--dataset_name', str(dataset_name),
        '--output_dir', str(output_dir),
        '--max_train_samples', str(int(max_train_samples)),
        '--max_eval_samples', str(int(max_eval_samples)),
        '--num_train_epochs', str(int(num_train_epochs)),
        '--per_device_train_batch_size', str(int(per_device_train_batch_size)),
        '--per_device_eval_batch_size', str(int(per_device_eval_batch_size)),
        '--gradient_accumulation_steps', str(int(gradient_accumulation_steps)),
    ]
    if extra_dataset_name:
        cmd.extend(['--extra_dataset_name', str(extra_dataset_name)])
    if dry_run:
        cmd.append('--dry_run')
    return cmd


def preview_command(*args):
    return ' '.join(build_command(*args))


def run_training(*args):
    global TRAIN_PROCESS
    if TRAIN_PROCESS is not None and TRAIN_PROCESS.poll() is None:
        yield '已有训练任务正在运行，请先停止当前任务。'
        return
    cmd = build_command(*args)
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    TRAIN_PROCESS = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env, text=True, encoding='utf-8', errors='replace')
    command_text = ' '.join(cmd)
    logs = ['启动命令:\n' + command_text + '\n', '\n===== 实时日志 =====\n']
    yield ''.join(logs)
    for line in TRAIN_PROCESS.stdout:
        logs.append(line)
        yield ''.join(logs)
    code = TRAIN_PROCESS.wait()
    logs.append('\n\n训练进程已结束，返回码: ' + str(code) + '\n')
    yield ''.join(logs)


def stop_training():
    global TRAIN_PROCESS
    if TRAIN_PROCESS is None or TRAIN_PROCESS.poll() is not None:
        return '当前没有正在运行的训练任务。'
    TRAIN_PROCESS.terminate()
    return '已请求停止训练进程。'


with gr.Blocks(title='PersonaCha Trainer') as demo:
    gr.Markdown('# PersonaCha 一键训练台\n\n目标是：少配置、少命令、少踩坑。先应用推荐预设，再做自检或正式训练。')

    with gr.Row():
        gr.Textbox(label='硬件检测', value=detect_hardware(), lines=8)
        gr.Markdown('## 当前设计原则\n\n- 默认适配 RTX 4060 8GB\n- 先做冒烟测试\n- 参数均有默认值\n- 日志实时显示')

    preset = gr.Dropdown(choices=['RTX 4060 8GB 推荐', '快速冒烟测试', '增强实验（含 Synthetic）'], value='RTX 4060 8GB 推荐', label='训练预设')
    apply_btn = gr.Button('应用推荐预设')

    with gr.Row():
        model_name = gr.Textbox(label='模型', value='facebook/blenderbot_small-90M')
        dataset_name = gr.Textbox(label='主数据集', value='bavard/personachat_truecased')
        extra_dataset_name = gr.Textbox(label='增强数据集（可空）', value='')

    with gr.Row():
        max_train_samples = gr.Number(label='训练样本数', value=5000, precision=0)
        max_eval_samples = gr.Number(label='验证样本数', value=500, precision=0)
        num_train_epochs = gr.Number(label='训练轮数', value=1, precision=0)

    with gr.Row():
        per_device_train_batch_size = gr.Number(label='训练 batch size', value=4, precision=0)
        per_device_eval_batch_size = gr.Number(label='验证 batch size', value=4, precision=0)
        gradient_accumulation_steps = gr.Number(label='梯度累积', value=4, precision=0)

    dry_run = gr.Checkbox(label='仅预处理自检（不正式训练）', value=True)

    with gr.Row():
        preview_btn = gr.Button('生成命令')
        start_btn = gr.Button('开始训练')
        stop_btn = gr.Button('停止训练')

    command_box = gr.Textbox(label='命令预览', lines=3)
    log_box = gr.Textbox(label='训练日志', lines=24)

    apply_btn.click(fn=apply_preset, inputs=preset, outputs=[model_name, dataset_name, extra_dataset_name, max_train_samples, max_eval_samples, num_train_epochs, per_device_train_batch_size, per_device_eval_batch_size, gradient_accumulation_steps])
    preview_btn.click(fn=preview_command, inputs=[model_name, dataset_name, extra_dataset_name, max_train_samples, max_eval_samples, num_train_epochs, per_device_train_batch_size, per_device_eval_batch_size, gradient_accumulation_steps, dry_run], outputs=command_box)
    start_btn.click(fn=run_training, inputs=[model_name, dataset_name, extra_dataset_name, max_train_samples, max_eval_samples, num_train_epochs, per_device_train_batch_size, per_device_eval_batch_size, gradient_accumulation_steps, dry_run], outputs=log_box)
    stop_btn.click(fn=stop_training, outputs=log_box)

if __name__ == '__main__':
    demo.launch(server_name='127.0.0.1', server_port=7860, inbrowser=True)
