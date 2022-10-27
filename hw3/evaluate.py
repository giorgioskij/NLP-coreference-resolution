import logging
from collections import Counter

from rich.progress import track
from copy import deepcopy

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

import argparse
import requests
import time

from requests.exceptions import ConnectionError
from typing import Tuple, List, Any, Dict


def flat_list(l: List[List[Any]]) -> List[Any]:
    return [_e for e in l for _e in e]


def count(l: List[Any]) -> Dict[Any, int]:
    d = {}
    for e in l:
        d[e] = 1 + d.get(e, 0)
    return d


def read_dataset(path: str) -> List[Dict]:
    samples: List[Dict] = []
    pron_counter = Counter()
    with open(path) as f:
        next(f)
        for line in f:
            (
                id,
                text,
                pron,
                p_offset,
                entity_A,
                offset_A,
                is_coref_A,
                entity_B,
                offset_B,
                is_coref_B,
                _,
            ) = line.strip().split("\t")
            pron_counter[pron.lower()] += 1
            samples.append({
                "id": id,
                "text": text,
                "pron": pron,
                "p_offset": int(p_offset),
                "entity_A": entity_A,
                "offset_A": int(offset_A),
                "is_coref_A": is_coref_A,
                "entity_B": entity_B,
                "offset_B": int(offset_B),
                "is_coref_B": is_coref_B,
            })
    print(pron_counter)
    return samples


def main(test_path: str, endpoint: str, batch_size=32):
    try:
        samples = read_dataset(test_path)
        prediction_samples = deepcopy(samples)
        for i, sample in enumerate(prediction_samples):
            del sample["is_coref_A"]
            del sample["is_coref_B"]
            prediction_samples[i] = sample

    except FileNotFoundError as e:
        logging.error(f"Evaluation crashed because {test_path} does not exist")
        exit(1)
    except Exception as e:
        logging.error(
            f"Evaluation crashed. Most likely, the file you gave is not in the correct format"
        )
        logging.error(f"Printing error found")
        logging.error(e, exc_info=True)
        exit(1)

    # TODO: change back to 10
    max_try = 10
    iterator = iter(range(max_try))

    while True:

        try:
            i = next(iterator)
        except StopIteration:
            logging.error(
                f"Impossible to establish a connection to the server even after 10 tries"
            )
            logging.error(
                "The server is not booting and, most likely, you have some error in build_model or StudentClass"
            )
            logging.error(
                "You can find more information inside logs/. Checkout both server.stdout and, most importantly, server.stderr"
            )
            exit(1)

        logging.info(
            f"Waiting 10 second for server to go up: trial {i}/{max_try}")
        time.sleep(10)

        try:
            response = requests.post(endpoint,
                                     json={
                                         "sentences": prediction_samples[:1]
                                     }).json()
            logging.info("Connection succeded")
            break
        except ConnectionError as e:
            continue
        except KeyError as e:
            logging.error(f"Server response in wrong format")
            logging.error(f"Response was: {response}")
            logging.error(e, exc_info=True)
            exit(1)

    predictions_123 = []
    predictions_23 = []
    predictions_3 = []

    for i in track(range(0, len(samples), batch_size),
                   description="Evaluating"):
        batch = prediction_samples[i:i + batch_size]
        try:
            response = requests.post(endpoint, json={"sentences": batch}).json()
            predictions_123 += response["predictions_123"]
            predictions_23 += response["predictions_23"]
            predictions_3 += response["predictions_3"]
        except KeyError as e:
            logging.error(f"Server response in wrong format")
            logging.error(f"Response was: {response}")
            logging.error(e, exc_info=True)
            exit(1)
    print("123 EVALUATION")
    evaluate(predictions_123, samples)
    print("23 EVALUATION")
    evaluate(predictions_23, samples)
    print("3 EVALUATION")
    evaluate(predictions_3, samples)


def evaluate(predictions_s, samples):
    total = 0
    correct = 0
    for pred, label in zip(predictions_s, samples):
        gold_pron_offset = label["p_offset"]
        pred_pron_offset = pred[0][1] if len(pred[0]) > 0 else None
        gold_pron = label["pron"]
        pred_pron = pred[0][0] if len(pred[0]) > 0 else None
        gold_both_wrong = label["is_coref_A"] == "FALSE" and label[
            "is_coref_B"] == "FALSE"
        pred_entity_offset = pred[1][1] if len(pred[1]) > 0 else None
        pred_entity = pred[1][0] if len(pred[1]) > 0 else None
        if gold_both_wrong:
            if pred_entity is None and gold_pron_offset == pred_pron_offset and gold_pron == pred_pron:
                correct += 1
            total += 1
        else:
            gold_entity_offset = (label["offset_A"] if label["is_coref_A"]
                                  == "TRUE" else label["offset_B"])
            gold_entity = (label["entity_A"] if label["is_coref_A"] == "TRUE"
                           else label["entity_B"])
            if (gold_pron_offset == pred_pron_offset and
                    gold_pron == pred_pron and
                    gold_entity_offset == pred_entity_offset and
                    gold_entity == pred_entity):
                correct += 1
            total += 1
    print(f"# instances: {total}")
    acc = float(correct) / total
    print(f"# accuracy: {acc:.4f}")


if __name__ == "__main__":
    # read_dataset("gap-train.tsv")
    parser = argparse.ArgumentParser()
    parser.add_argument("file",
                        type=str,
                        help="File containing data you want to evaluate upon")
    args = parser.parse_args()

    main(test_path=args.file, endpoint="http://127.0.0.1:12345")
