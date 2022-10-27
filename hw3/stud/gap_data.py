"""
Handles data-related stuff.
"""

#%% imports
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

from transformers import BatchEncoding

from evaluate import read_dataset
# from . import config
# from .ner_tagger import NerModel, Vocabulary, ner_predict

import torchcrf

import pytorch_lightning as pl
import torch
import torch.utils.data as torchdata

from transformers.tokenization_utils import PreTrainedTokenizer

#%% datasets


# datamodule that extends pl.LightningDataModule and uses read_dataset
# to load data
class GapDatamodule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir: Path,
        tokenizer: PreTrainedTokenizer,
        upsample_neither: bool = False,
        batch_size: int = 32,
        show_candidates: bool = True,
        device: Optional[torch.device] = None,
        max_len: int = 385,
    ):
        super().__init__()
        self.data_dir: Path = data_dir
        self.batch_size: int = batch_size
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.upsample_neither: bool = upsample_neither
        self.show_candidates: bool = show_candidates

        self.device: torch.device = device or torch.device(
            'cuda' if torch.cuda.is_available else 'cpu')

        self.train_data: Optional[GapDataset] = None
        self.valid_data: Optional[GapDataset] = None
        self.max_len: int = max_len
        self.crf: torchcrf.CRF

    def collate_fn(self, batch):
        # return lambda b: fn(b, tokenizer)

        if self.show_candidates:
            return collate_fn_candidates(batch, self.tokenizer)

        return collate_fn_no_candidates(batch, self.tokenizer, self.max_len)

    def setup(self, stage: Optional[str] = None):
        if self.train_data is not None and self.valid_data is not None:
            return

        train_sentences = read_dataset(str(self.data_dir / 'train.tsv'))

        if self.upsample_neither:
            neither_sentences = read_dataset(str(self.data_dir / 'neither.tsv'))
            train_sentences += neither_sentences

        self.train_data = GapDataset(
            data=train_sentences,
            tokenizer=self.tokenizer,
            show_candidates=self.show_candidates,
        )

        self.valid_data = GapDataset(
            data=read_dataset(str(self.data_dir / 'dev.tsv')),
            tokenizer=self.tokenizer,
            show_candidates=self.show_candidates,
        )

        # self.max_len = max(self.train_data.max_len, self.valid_data.max_len)

    def train_dataloader(self):
        if self.train_data is None:
            raise ValueError('train_data is None')
        return torchdata.DataLoader(self.train_data,
                                    batch_size=self.batch_size,
                                    collate_fn=self.collate_fn,
                                    num_workers=20,
                                    shuffle=True)

    def val_dataloader(self):
        if self.valid_data is None:
            raise ValueError('valid_data is None')
        return torchdata.DataLoader(self.valid_data,
                                    batch_size=self.batch_size,
                                    collate_fn=self.collate_fn,
                                    num_workers=20)


class GapDataset(torchdata.Dataset):

    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        show_candidates: bool = True,
    ):
        self.data: List[Dict] = data
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.show_candidates: bool = show_candidates

        # self.max_len: int = 0
        # for d in self.data:
        #     t = self.tokenizer(d['text'], add_special_tokens=False)
        #     self.max_len = max(self.max_len, len(t.input_ids))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[str, Union[int, Tuple], int]:
        return process_sentence(self.data[idx], self.show_candidates)


def process_sentence(
        sentence_dict: Dict,
        show_candidates: bool = True,
        hide_labels: bool = False
) -> Tuple[str, Optional[Union[int, Tuple]], int]:
    if hide_labels:
        (sentence_id, text, pron, p_offset, a, a_offset, b,
         b_offset) = sentence_dict.values()
    else:
        (sentence_id, text, pron, p_offset, a, a_offset, a_correct, b, b_offset,
         b_correct) = sentence_dict.values()
        a_correct = a_correct.lower() != 'false'
        b_correct = b_correct.lower() != 'false'

    # add special tokens to the sentence
    if show_candidates:
        words = [pron, a, b]
        separators = ['[P]', '[A]', '[B]']
        offsets = [p_offset, a_offset, b_offset]
    else:
        words = [pron]
        separators = ['[P]']
        offsets = [p_offset]

    order = torch.argsort(torch.tensor(offsets), descending=True).tolist()

    # insert special tokens [P], [A], [B]
    for i in order:
        word = words[i]
        offset = offsets[i]
        sep = separators[i]
        text = (text[:offset] + sep + ' ' + word + ' ' + sep +
                text[offset + len(word):])

    # compute labels
    if hide_labels:
        return text, None, sentence_id

    if show_candidates:
        # if A and B are marked in the sentence, label is simply ternary
        # 0: neither, 1: A, 2: B
        label: int = 0
        if a_correct:
            label = 1
        if b_correct:
            label = 2
        return text, label, sentence_id
    else:
        # if candidates are not marked, label is a tuple
        # <offset of the coreference>, <text of the coreference>
        correct, offset = None, None
        if a_correct:
            correct = a
            offset = a_offset
        if b_correct:
            correct = b
            offset = b_offset

        if offset is not None and p_offset < offset:
            offset += 8
        return text, (correct, offset), sentence_id


def word2indices(encoded, tokenizer):
    l = []
    for i in range(len(encoded.input_ids)):
        # we have to collect all the subtokenizations of the words
        w_id2input_ids = dict()
        # we have to keep track of the indices that compose the word
        # so we can use them to extract the embeddings
        w_id2_indices = dict()
        for idx, (w_id, input_id) in enumerate(
                zip(encoded.word_ids(i), encoded.input_ids[i])):
            w_id2input_ids.setdefault(w_id, []).append(input_id)
            w_id2_indices.setdefault(w_id, []).append(idx)

        # now let's decode all the words that we have
        word2_indices = dict()
        for w_id, input_ids in w_id2input_ids.items():
            word = tokenizer.decode(input_ids)
            word2_indices[word] = w_id2_indices[w_id]

        l.append(word2_indices)
    return l


def collate_fn_no_candidates(batch: List[Tuple], tokenizer, max_len):
    sentences, labels, _ = zip(*batch)

    t = tokenizer(
        list(sentences),
        add_special_tokens=False,
        return_tensors='pt',
        padding='max_length',
        max_length=max_len,
    )

    ids = t.input_ids

    p_encoding = tokenizer.encode(
        '[P]',
        add_special_tokens=False,
    )[0]

    # this tensor will contain the label for each token as:
    # 0: nothing; 1: pronoun; 2: B-coref; 3: I-coref; -1: padding
    new_labels = torch.zeros_like(t['input_ids'])

    # find the vectors coming from the pronoun
    p_indices = (ids == p_encoding).nonzero(as_tuple=True)[1][::2] + 1

    if not all(labels):
        return (ids, t['attention_mask'], None, None, p_indices, None, None)

    # set the label of the pronouns to 1
    new_labels[torch.arange(new_labels.shape[0]), p_indices] = 1

    # set the label of the padding to -1
    new_labels[ids == 0] = -1

    # for each sentence, find the tokens coming from the correct coreference
    for i, (sentence, (coref, offset)) in enumerate(zip(sentences, labels)):
        if coref is None and offset is None:
            continue
        half_sentence = sentence[offset:]
        half_t = tokenizer.encode(half_sentence,
                                  add_special_tokens=False,
                                  return_tensors='pt')[0]
        for j in range(len(ids[i])):
            if ids[i, j:j + len(half_t)].equal(half_t):
                first_coref_vector_position = j
                break
        else:
            raise ValueError('something is not right with the sentence')

        coref_t = tokenizer.encode(coref, add_special_tokens=False)
        last_coref_vector_position = (first_coref_vector_position +
                                      len(coref_t) - 1)

        # set the label of B-coref to 2, and of I-coref to 3
        new_labels[i][first_coref_vector_position] = 2
        new_labels[i][first_coref_vector_position +
                      1:last_coref_vector_position + 1] = 3

    b_positions = []
    for s in range(len(new_labels)):
        if (new_labels[s] == 2).sum().item() == 0:
            b_positions.append(max_len)
        else:
            b_positions.append((new_labels[s] == 2).nonzero().item())
    b_positions = torch.tensor(b_positions)

    return (ids, t['attention_mask'], b_positions, new_labels, p_indices, None,
            None)


def collate_fn_candidates(batch: List[Tuple[str, int, int]],
                          tokenizer,
                          hide_labels: bool = False):
    sentences, labels, _ = zip(*batch)
    # if all(labels):
    if not hide_labels:
        labels = torch.tensor(labels, dtype=torch.long)
    t = tokenizer(
        list(sentences),  # type: ignore
        add_special_tokens=False,
        return_tensors='pt',
        padding=True)

    # after tokenization, find indices of pronouns in each sentence
    p_encoding, a_encoding, b_encoding = tokenizer.encode(
        '[P] [A] [B]',
        add_special_tokens=False,
    )

    # since the pronouns are always very short, we are guaranteed that
    # they will never get split by the tokenizer
    p_indices = (t.input_ids == p_encoding).nonzero(as_tuple=True)[1][::2]

    # now p_indices contains the index of the first [P] in each sentence
    # add 1 to get the index of the actual pronoun
    p_indices += 1

    # these two contain the indices of the markers [A] and [B] in each sentence
    a_marker_indices = (t.input_ids == a_encoding).nonzero(as_tuple=True)[1]
    b_marker_indices = (t.input_ids == b_encoding).nonzero(as_tuple=True)[1]

    return t['input_ids'], t[
        'attention_mask'], labels, p_indices, a_marker_indices, b_marker_indices
