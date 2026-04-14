#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name', type=str, default='facebook/blenderbot_small-90M')
    p.add_argument('--dataset_name', type=str, default='bavard/personachat_truecased')
    p.add_argument('--extra_dataset_name', type=str, default='')
    p.add_argument('--train_split', type=str, default='train')
    p.add_argument('--eval_split', type=str, default='validation')
    p.add_argument('--output_dir', type=str, default='./experiments/outputs/blenderbot_personachat')
    p.add_argument('--max_source_length', type=int, default=256)
    p.add_argument('--max_target_length', type=int, default=64)
    p.add_argument('--max_train_samples', type=int, default=5000)
    p.add_argument('--max_eval_samples', type=int, default=500)
    p.add_argument('--num_train_epochs', type=int, default=1)
    p.add_argument('--per_device_train_batch_size', type=int, default=4)
    p.add_argument('--per_device_eval_batch_size', type=int, default=4)
    p.add_argument('--learning_rate', type=float, default=5e-5)
    p.add_argument('--gradient_accumulation_steps', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save_steps', type=int, default=500)
    p.add_argument('--logging_steps', type=int, default=50)
    p.add_argument('--dry_run', action='store_true')
    return p.parse_args()


def join_history(history):
    turns = [str(x).strip() for x in history if str(x).strip()]
    if not turns:
        return ''
    rows = []
    for i, text in enumerate(turns):
        speaker = 'User' if i % 2 == 0 else 'Bot'
        rows.append(f'{speaker}: {text}')
    return '\n'.join(rows)


def build_source(persona_lines, history):
    persona_lines = [str(x).strip() for x in persona_lines if str(x).strip()]
    persona_block = '\n'.join(f'- {x}' for x in persona_lines) if persona_lines else '- unknown persona'
    history_block = join_history(history) or 'User: hello'
    return 'Persona:\n' + persona_block + '\n\nDialogue history:\n' + history_block + '\n\nGenerate the next bot response:'


def record_to_rows(record: dict[str, Any]):
    rows = []
    if 'utterances' in record and isinstance(record['utterances'], list):
        personality = record.get('personality', []) or record.get('persona', []) or []
        for utt in record['utterances']:
            if not isinstance(utt, dict):
                continue
            history = utt.get('history', []) or []
            response = ''
            if isinstance(utt.get('response'), str):
                response = utt['response'].strip()
            elif isinstance(utt.get('candidates'), list) and utt['candidates']:
                response = str(utt['candidates'][-1]).strip()
            if history and response:
                rows.append({'source': build_source(personality, history), 'target': response})
        return rows

    persona_keys = ['persona', 'personas', 'profile', 'user_profile', 'personality']
    history_keys = ['history', 'context', 'dialogue_history', 'conversation', 'input']
    target_keys = ['response', 'target', 'output', 'answer', 'label']

    persona = []
    history = []
    target = ''

    for key in persona_keys:
        if key in record:
            value = record[key]
            if isinstance(value, list):
                persona = [str(x) for x in value]
            elif isinstance(value, str):
                persona = [x.strip() for x in value.split('\n') if x.strip()]
            break

    for key in history_keys:
        if key in record:
            value = record[key]
            if isinstance(value, list):
                history = [str(x) for x in value]
            elif isinstance(value, str):
                history = [x.strip() for x in value.split('\n') if x.strip()]
            break

    for key in target_keys:
        if isinstance(record.get(key), str):
            target = record[key].strip()
            break

    if history and target:
        rows.append({'source': build_source(persona, history), 'target': target})
    return rows


def flatten_split(split, max_rows: int):
    rows = []
    for record in split:
        for row in record_to_rows(record):
            if row['source'].strip() and row['target'].strip():
                rows.append(row)
                if max_rows and len(rows) >= max_rows:
                    return Dataset.from_list(rows)
    return Dataset.from_list(rows)


def choose_split_name(raw, preferred, fallback):
    names = set(raw.keys())
    if preferred in names:
        return preferred
    if fallback in names:
        return fallback
    return list(names)[0]


def load_and_prepare(dataset_name, train_split, eval_split, max_train, max_eval):
    raw = load_dataset(dataset_name)
    train_name = choose_split_name(raw, train_split, 'train')
    eval_name = choose_split_name(raw, eval_split, 'validation')
    train_ds = flatten_split(raw[train_name], max_train)
    eval_ds = flatten_split(raw[eval_name], max_eval)
    return train_ds, eval_ds


def main():
    args = parse_args()
    set_seed(args.seed)

    train_ds, eval_ds = load_and_prepare(args.dataset_name, args.train_split, args.eval_split, args.max_train_samples, args.max_eval_samples)

    if args.extra_dataset_name:
        extra_train, extra_eval = load_and_prepare(args.extra_dataset_name, args.train_split, args.eval_split, max(1, args.max_train_samples // 2), max(1, args.max_eval_samples // 2))
        if len(extra_train) > 0:
            train_ds = concatenate_datasets([train_ds, extra_train])
        if len(extra_eval) > 0:
            eval_ds = concatenate_datasets([eval_ds, extra_eval])

    if len(train_ds) == 0:
        raise RuntimeError('No valid training rows were extracted from the dataset.')
    if len(eval_ds) == 0:
        eval_ds = train_ds.select(range(min(100, len(train_ds))))

    print('Prepared datasets:')
    print('train:', len(train_ds), 'eval:', len(eval_ds))
    print('example source:\n', train_ds[0]['source'])
    print('example target:\n', train_ds[0]['target'])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    def preprocess(batch):
        model_inputs = tokenizer(batch['source'], max_length=args.max_source_length, truncation=True, padding=False)
        labels = tokenizer(text_target=batch['target'], max_length=args.max_target_length, truncation=True, padding=False)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    tokenized_train = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    tokenized_eval = eval_ds.map(preprocess, batched=True, remove_columns=eval_ds.column_names)

    if args.dry_run:
        print('Dry run complete. Exiting before training.')
        return

    os.makedirs(args.output_dir, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy='steps',
        save_strategy='steps',
        logging_strategy='steps',
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        logging_steps=args.logging_steps,
        predict_with_generate=True,
        bf16=False,
        fp16=False,
        report_to='none',
        load_best_model_at_end=False,
        remove_unused_columns=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    prompts = [
        'Persona:\n- I like dogs\n- I play guitar\n\nDialogue history:\nUser: hi, what do you do for fun?\n\nGenerate the next bot response:',
        'Persona:\n- I love hiking\n- I work as a teacher\n\nDialogue history:\nUser: nice to meet you. what do you enjoy on weekends?\n\nGenerate the next bot response:',
    ]
    for i, prompt in enumerate(prompts, 1):
        inputs = tokenizer(prompt, return_tensors='pt')
        outputs = model.generate(**inputs, max_new_tokens=40)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f'\n=== sample generation {i} ===')
        print(text)


if __name__ == '__main__':
    main()
