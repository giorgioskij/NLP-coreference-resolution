import argparse
from typing import List, Dict

from hw3.evaluate import read_dataset
from stud.implementation import build_model_3


def main(sentences: List[Dict]):

    model = build_model_3("cpu")
    predicted_sentences = model.predict(sentences)

    for i, (sentence, ((pron, pron_offset), (entity, entity_offset))) in enumerate(
        zip(sentences, predicted_sentences)
    ):
        print(f"# sentence = {sentence['text']}")
        print(f"{i}\t{pron}\t{pron_offset}\t{entity}\t{entity_offset}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file", type=str, help="File containing data you want to evaluate upon"
    )
    args = parser.parse_args()
    sentences = read_dataset(args.file)
    main(sentences=sentences)
