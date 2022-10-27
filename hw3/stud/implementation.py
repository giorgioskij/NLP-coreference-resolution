import re

import numpy as np
from typing import List, Tuple, Dict
# import config
from model import Model

# print(os.getcwd())
# print(os.listdir())
# os.chdir('hw3')
# print(os.listdir())
from stud import gap_model, gap_data
# import gap_mode
# import gap_data


def build_model_123(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 1, 2 and 3 of the Coreference resolution pipeline.
            1: Ambiguous pronoun identification.
            2: Entity identification
            3: Coreference resolution
    """
    return RandomBaseline(True, True)


def build_model_23(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 2 and 3 of the Coreference resolution pipeline.
            2: Entity identification
            3: Coreference resolution
    """
    return RandomBaseline(False, True)
    # return StudentModel23(device)


def build_model_3(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements step 3 of the Coreference resolution pipeline.
            3: Coreference resolution
    """
    return StudentModel3(device)
    # return RandomBaseline(False, False)


class RandomBaseline(Model):

    def __init__(self, predict_pronoun: bool, predict_entities: bool):
        self.pronouns_weights = {
            "his": 904,
            "her": 773,
            "he": 610,
            "she": 555,
            "him": 157,
        }
        self.predict_pronoun = predict_pronoun
        self.pred_entities = predict_entities

    def predict(
            self, sentences: List[Dict]
    ) -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:
        predictions = []
        for sent in sentences:
            text = sent["text"]
            toks = re.sub("[.,'`()]", " ", text).split(" ")
            if self.predict_pronoun:
                prons = [
                    tok.lower()
                    for tok in toks
                    if tok.lower() in self.pronouns_weights
                ]
                if prons:
                    pron = np.random.choice(prons, 1, self.pronouns_weights)[0]
                    pron_offset = text.lower().index(pron)
                    if self.pred_entities:
                        entity = self.predict_entity(predictions, pron,
                                                     pron_offset, text, toks)
                    else:
                        entities = [sent["entity_A"], sent["entity_B"]]
                        entity = self.predict_entity(predictions, pron,
                                                     pron_offset, text, toks,
                                                     entities)
                    predictions.append(((pron, pron_offset), entity))
                else:
                    predictions.append(((), ()))
            else:
                pron = sent["pron"]
                pron_offset = sent["p_offset"]
                if self.pred_entities:
                    entity = self.predict_entity(predictions, pron, pron_offset,
                                                 text, toks)
                else:
                    entities = [
                        (sent["entity_A"], sent["offset_A"]),
                        (sent["entity_B"], sent["offset_B"]),
                    ]
                    entity = self.predict_entity(predictions, pron, pron_offset,
                                                 text, toks, entities)
                predictions.append(((pron, pron_offset), entity))
        return predictions

    def predict_entity(self,
                       predictions,
                       pron,
                       pron_offset,
                       text,
                       toks,
                       entities=None):
        entities = (entities if entities is not None else self.predict_entities(
            entities, toks))
        entity_idx = np.random.choice([0, len(entities) - 1], 1)[0]
        return entities[entity_idx]

    def predict_entities(self, entities, toks):
        offset = 0
        entities = []
        for tok in toks:
            if tok != "" and tok[0].isupper():
                entities.append((tok, offset))
            offset += len(tok) + 1
        return entities


class StudentModel3(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def __init__(self, device: str):

        self.model = None
        self.device = device

    def predict(self, sentences: List[Dict]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        if self.model is None:
            self.model: gap_model.GapModel = gap_model.GapModel.load_from_checkpoint(
                "model/distilbert-lstm-83.ckpt", map_location=self.device)
            self.model.eval()

        outputs = []
        for s in sentences:
            # prediction = self.model.predict_sentence_candidates(s["text"])
            sentence_data = gap_data.process_sentence(s,
                                                      show_candidates=True,
                                                      hide_labels=True)
            single_sentence_batch = gap_data.collate_fn_candidates(
                [sentence_data], self.model.tokenizer,
                hide_labels=True)  # type: ignore

            ids, mask, _, p, a, b = single_sentence_batch
            _, logits, _ = self.model(ids, mask, p, a, b)
            prediction = logits.argmax(dim=-1).item()

            predicted_entity = []
            if prediction == 1:
                predicted_entity = s["entity_A"], s["offset_A"]
            if prediction == 2:
                predicted_entity = s["entity_B"], s["offset_B"]
            output = ((s["pron"], s["p_offset"]), predicted_entity)
            outputs.append(output)
        return outputs


class StudentModel23(Model):

    def __init__(self, device: str):

        self.model: gap_model.GapModel = gap_model.GapModel.load_from_checkpoint(
            "model/distilbert23.ckpt",
            ckp_id="model/distilbert23",
            nertagger_ckp_id="model/nertagger",
            map_location=device)
        self.model.eval()

    def predict(self, sentences: List[Dict]) -> List[List[str]]:

        if self.model is None:
            self.model: gap_model.GapModel = gap_model.GapModel.load_from_checkpoint(
                "model/distilbert23.ckpt", map_location=self.device)
            self.model.eval()

        outputs = []
        for s in sentences:
            sentence_data = gap_data.process_sentence(s,
                                                      show_candidates=False,
                                                      hide_labels=True)
            single_sentence_batch = gap_data.collate_fn_no_candidates(
                [sentence_data], self.model.tokenizer,
                max_len=400)  # type: ignore
            ids, mask, _, _, p, a, b = single_sentence_batch
            _, _, predictions = self.model(ids, mask, p, a, b)

            # from the outputs of the model, extract back the coreference
            # text and offset
            text = sentence_data[0]
            prediction = []
            predicted_entity_idx = (predictions >= 2).nonzero(
                as_tuple=True)[1].flatten().tolist()
            if len(predicted_entity_idx):
                e = self.model.tokenizer(text, add_special_tokens=False)
                start_offset = e.token_to_chars(predicted_entity_idx[0]).start
                end_offset = e.token_to_chars(predicted_entity_idx[-1]).end
                coref = text[start_offset:end_offset]

                if s["p_offset"] < start_offset:
                    start_offset -= 8

                prediction = (coref, start_offset)

            output = ((s["pron"], s["p_offset"]), prediction)
            outputs.append(output)

        return outputs
