import os
import json
import math
import random
import argparse
from pathlib import Path

import tqdm


random.seed(42)


def read_dataset(base_data_path: Path):
    data_paths = base_data_path.glob("*.json")
    dataset = []
    for path in data_paths:
        with open(path, "r", encoding="utf-8") as f:
            dataset.extend(json.load(f))

    return dataset


def flat_dataset(dataset):
    new_dataset = []
    for sample in tqdm.tqdm(dataset):
        chapters = sample["chapters"]
        new_dataset.extend(chapters)
    return new_dataset


def write_dataset(path, data):
    print(f"Write data to {path}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", type=Path, default=Path("./data/data_example"))
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--train_test_dir", type=Path, default=Path("./data/train_test"))

    args = parser.parse_args()

    dataset = read_dataset(args.raw_data_dir)

    test_size = math.floor(args.test_size * len(dataset))
    train_size = len(dataset) - test_size

    print(f"Train books: {train_size}, test books: {test_size}")

    train_books = dataset[:train_size]
    test_books = dataset[train_size:]

    assert len(train_books) == train_size and len(test_books) == test_size, (
        len(train_books),
        train_size,
        len(test_books),
        test_books,
    )

    train, test = flat_dataset(train_books), flat_dataset(test_books)
    print(f"Train size: {len(train)}, test size: {len(test)}")

    if not os.path.exists(args.train_test_dir):
        os.mkdir(args.train_test_dir)

    print(f"Save data to {args.train_test_dir}")
    write_dataset(args.train_test_dir / "train.json", train)
    write_dataset(args.train_test_dir / "validation.json", test)
