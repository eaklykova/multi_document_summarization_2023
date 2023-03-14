import os
import re
import json
import uuid
import argparse
from pathlib import Path

import tqdm
from rouge import Rouge
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def WHITESPACE_HANDLER(k):
    return re.sub("\s+", " ", re.sub("\n+", " ", k.strip()))


def generate_summ(model, tokenizer, text):
    input_ids = tokenizer(
        [WHITESPACE_HANDLER(text)],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )["input_ids"]
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


def validate(model, tokenizer, dataset):
    refs = []
    hyps = []
    raw_text = []
    rouge = Rouge()
    for sample in tqdm.tqdm(dataset):
        refs.append(sample["chapter_summary"])
        try:
            raw_text.append(sample["chapters_text"])
        except Exception:
            print(sample)
            exit(0)
        hyps.append(generate_summ(model, tokenizer, text=sample["chapters_text"]))
    scores = rouge.get_scores(hyps, refs, avg=True)
    return scores, hyps, refs, raw_text


def write_scores(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="csebuetnlp/mT5_multilingual_XLSum")
    parser.add_argument("--data_path", type=Path, default=Path("./data/data_example/corrected_alina_12.json"))
    parser.add_argument("--val_results_path", type=Path, default=Path("./eval_logs/"))
    parser.add_argument("--validate_size", default=-1, type=int)
    parser.add_argument("--is_processed_data", action="store_true", dest="is_processed_data")
    parser.set_defaults(is_processed_data=False)

    args = parser.parse_args()

    dataset = read_dataset(args.data_path)

    if not args.is_processed_data:
        dataset = flat_dataset(dataset)
    validate_size = args.validate_size
    assert validate_size >= -1
    if validate_size == -1:
        validate_size = len(dataset)
    dataset = dataset[:validate_size]
    print(f"Validate samples: {len(dataset)}")

    model_name = args.model_name

    print(f"Validate {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if not os.path.exists(args.val_results_path):
        os.mkdir(args.val_results_path)

    scores, hyps, refs, raw_text = validate(model, tokenizer, dataset)
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
