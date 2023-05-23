import os
import re
import json
import uuid
import argparse
from pathlib import Path

import torch
import tqdm
import nltk
import datasets
from rouge import Rouge
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.text_rank import TextRank


def WHITESPACE_HANDLER(k):
    return re.sub("\s+", " ", re.sub("\n+", " ", k.strip()))


def generate_summ(model, tokenizer, device, text):
    input_ids = tokenizer(
        [WHITESPACE_HANDLER("Summarize: " + text)],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )["input_ids"].to(device)
    output_ids = model.generate(input_ids=input_ids, max_length=256, no_repeat_ngram_size=2, num_beams=4)[0]
    summary = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return summary


def read_dataset(base_data_path):
    with open(base_data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset


def flat_dataset(dataset):
    new_dataset = []
    for sample in tqdm.tqdm(dataset):
        chapters = sample["chapters"]
        new_dataset.extend(chapters)
    return new_dataset


def validate(model, tokenizer, dataset, device, eval_type, chapter_summary_name='chapter_summary'):
    refs = []
    hyps = []
    raw_text = []
    rouge = Rouge()
    for sample in tqdm.tqdm(dataset):
        refs.append(sample[chapter_summary_name])
        try:
            raw_text.append(sample["chapters_text"])
        except Exception:
            print(sample)
            exit(0)
        if eval_type == 'hierarchical':
            hyps.append(hierarchical_summarization(model, tokenizer, device, text=sample["chapters_text"]))
        elif eval_type == 'baseline':
            hyps.append(generate_summ(model, tokenizer, device, text=sample["chapters_text"]))
        elif eval_type == 'text_rank':
            hyps.append(text_rank_summarization(model, text=sample["chapters_text"]))
        elif eval_type == 'complex_1':
            hyps.append(text_rank_seq_2_seq_summarization(model, tokenizer, device, text=sample["chapters_text"]))
        else:
            raise NotImplementedError(f'eval_type "{eval_type}" not implemented')
    scores = rouge.get_scores(hyps, refs, avg=True)
    return scores, hyps, refs, raw_text


def hierarchical_summarization(model, tokenizer, device, text):
    sentences = nltk.sent_tokenize(text, language="russian")
    summ = sentences[:]
    merge_cnt = 10
    is_summarized = False
    while not(len(summ) <= 4 and is_summarized):
        new_summ = []
        for ind in range(0, len(summ), merge_cnt):
            pairs = summ[ind:ind + merge_cnt]
            pairs = ' '.join(pairs)
            new_summ.append(generate_summ(model, tokenizer, device, pairs))
        summ = new_summ[:]
        is_summarized = True
    return ' '.join(summ)


def text_rank_summarization(model, text):
    summary_raw = model(text, ratio=0.16)
    summary = ' '.join([t['sentence'] for t in summary_raw])
    return summary


def text_rank_seq_2_seq_summarization(model, tokenizer, device, text):
    seq2seq_model, text_rank = model
    text_rank_summary = text_rank_summarization(text_rank, text)
    res = hierarchical_summarization(seq2seq_model, tokenizer, device, text_rank_summary)
    return res


def write_scores(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="csebuetnlp/mT5_multilingual_XLSum")
    parser.add_argument("--data_path", type=str, default="./data/data_example/corrected_alina_12.json")
    parser.add_argument("--val_results_path", type=Path, default=Path("./eval_logs/"))
    parser.add_argument("--validate_size", default=-1, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--is_processed_data", action="store_true", dest="is_processed_data")
    parser.add_argument("--is_hf_dataset", action="store_true", dest="is_hf_dataset")
    parser.add_argument("--eval_type", choices=['baseline', 'hierarchical', 'text_rank', 'complex_1'], default='baseline')
    parser.add_argument("--chapter_summary_name", type=str, default='chapter_summary')
    parser.set_defaults(is_processed_data=False, is_hf_dataset=False)

    args = parser.parse_args()

    validate_size = args.validate_size

    device = torch.device(args.device)

    if args.is_hf_dataset:
        dataset = datasets.load_dataset(args.data_path, split='validation')
        assert validate_size >= -1
        if validate_size == -1:
            validate_size = len(dataset)
        dataset = dataset.select(range(validate_size))
    else:
        dataset = read_dataset(args.data_path)

        if not args.is_processed_data:
            dataset = flat_dataset(dataset)
        assert validate_size >= -1
        if validate_size == -1:
            validate_size = len(dataset)
        dataset = dataset[:validate_size]
    print(f"Validate samples: {len(dataset)}")

    model_name = args.model_name

    print(f"Validate {model_name}")

    if model_name == 'text_rank':
        tokenizer = None
        model = TextRank()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        model.to(device)
    
    if args.eval_type == 'complex_1':
        model = (model, TextRank())

    if not os.path.exists(args.val_results_path):
        os.mkdir(args.val_results_path)

    scores, hyps, refs, raw_text = validate(
        model,
        tokenizer,
        dataset,
        device,
        eval_type=args.eval_type,
        chapter_summary_name=args.chapter_summary_name
    )
    write_scores(
        args.val_results_path / f"{model_name.replace('/', '_')}_samples_{len(dataset)}_{uuid.uuid4()}.json", scores
    )
    print("Results:")
    print(scores)
    print("========================================================")
    print("Golden summary:\n========================================================")
    print(refs[0])
    print("========================================================\nExample summary:")
    print(hyps[0])
    print("========================================================\nText:")
    print(raw_text[0])
