from pathlib import Path
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple
from torch import Tensor
from torch.nn import functional
import torch
import torchmetrics
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForTokenClassification
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from stud.gap_data import GapDataset

from pytorch_pretrained_bert.optimization import BertAdam


class GapModel(pl.LightningModule):

    def __init__(
        self,
        lr: float = 1e-3,
        batch_size: int = 64,
        freeze_bert: bool = True,
        ckp_id: str = 'distilbert-base-cased',
        use_last_n_layers: int = 1,
        double_linear: bool = False,
        clf_hidden_dim: int = 1024,
        clf_dropout: float = 0.5,
        features_dropout: float = 0.5,
        bert_dropout: float = 0.1,
        use_lstm: bool = False,
        lstm_hidden_dim: int = 1024,
        bert_reduction: str = 'cat',
        uniform_weight_decay: bool = False,
        label_smoothing: float = 0.0,
        clf_weight_decay: Optional[float] = None,
        lstm_n_layers: int = 1,
        use_lstm_last_out: bool = False,
        show_candidates: bool = True,
        nertagger_ckp_id:
        str = 'dbmdz/bert-large-cased-finetuned-conll03-english',
        max_len: int = 400,
    ):
        super().__init__()

        # hyperparameters
        self.save_hyperparameters()
        self.ckp_id: str = ckp_id
        self.lr: float = lr
        self.batch_size: int = batch_size
        self.use_last_n_layers: int = use_last_n_layers
        self.double_linear: bool = double_linear
        self.clf_hidden_dim: int = clf_hidden_dim
        self.use_lstm: bool = use_lstm
        self.lstm_hidden_dim: int = lstm_hidden_dim
        self.features_dropout_p: float = features_dropout
        self.bert_dropout: float = bert_dropout
        if not bert_reduction in {'cat', 'mean', 'sum'}:
            raise ValueError(
                f"Wrong reduction strategy: {bert_reduction}: choose between \
                    'cat', 'mean' or 'sum'")
        self.bert_reduction: str = bert_reduction
        self.uniform_weight_decay: bool = uniform_weight_decay
        self.label_smoothing: float = label_smoothing
        self.clf_weight_decay: Optional[float] = clf_weight_decay
        self.lstm_n_layers: int = lstm_n_layers
        self.use_lstm_last_out: bool = use_lstm_last_out
        self.show_candidates: bool = show_candidates
        self.max_len: int = max_len
        self.num_classes: int = 3 if self.show_candidates else self.max_len + 1
        self.nertagger_ckp_id: str = nertagger_ckp_id

        # BERT
        configuration = AutoConfig.from_pretrained(self.ckp_id)
        configuration.hidden_dropout_prob = self.bert_dropout
        configuration.attention_probs_dropout_prob = self.bert_dropout
        self.bert = AutoModel.from_pretrained(self.ckp_id, config=configuration)
        # add special tokens and freeze
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(  # type: ignore
            self.ckp_id)
        self.tokenizer.add_tokens(['[P]', '[A]', '[B]'])
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.bert_out_size: int = self.bert.config.hidden_size
        if self.bert_reduction == 'cat':
            self.bert_out_size *= self.use_last_n_layers

        # LSTM
        self.feature_out_size: int = self.bert_out_size
        if self.use_lstm:
            self.lstm = torch.nn.LSTM(
                input_size=self.bert_out_size,
                hidden_size=self.lstm_hidden_dim,
                num_layers=self.lstm_n_layers,
                bidirectional=self.show_candidates,
                batch_first=True,
                dropout=0.5,
            )
            self.feature_out_size = self.lstm_hidden_dim
            if self.show_candidates:
                self.feature_out_size *= 2

        # classifier takes the concatenation of 3 vectors (p, a, b)
        if self.show_candidates:
            num_vectors = 4 if self.use_lstm_last_out else 3
            clf_features_in = self.feature_out_size * num_vectors
        else:
            clf_features_in = self.feature_out_size * self.max_len

        if not self.show_candidates:
            self.nertagger = AutoModelForTokenClassification.from_pretrained(
                self.nertagger_ckp_id)
            for param in self.nertagger.parameters():
                param.requires_grad = False
            self.nertagger.resize_token_embeddings(len(self.tokenizer))
            # clf_features_in += len(self.nertagger.config.id2label)

        if self.double_linear:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(clf_features_in, self.clf_hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(clf_dropout),
                torch.nn.Linear(self.clf_hidden_dim, self.num_classes),
            )
        else:
            self.classifier = torch.nn.Linear(clf_features_in, self.num_classes)

        # dropout
        if self.features_dropout_p > 0:
            self.features_dropout = torch.nn.Dropout(self.features_dropout_p)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # metrics
        self.loss_fn = torch.nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing,
            # ignore_index=-1,
        )
        self.train_acc = torchmetrics.Accuracy(ignore_index=-1,
                                               compute_on_cpu=True)
        self.valid_acc = torchmetrics.Accuracy(ignore_index=-1,
                                               compute_on_cpu=True)
        self.test_acc = torchmetrics.Accuracy(ignore_index=-1,
                                              compute_on_cpu=True)
        # self.valid_f1 = torchmetrics.F1Score(average='macro',
        #                                      num_classes=self.num_classes,
        #                                      ignore_index=-1)
        self.total_train = self.total_valid = 0
        self.correct_train = self.correct_valid = 0
        self.entities_correct = self.entities_total = 0
        self.count_pron_before = 0
        self.n_bper_per_sentence = []

    def forward_no_candidates(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pron_positions: Tensor,
        labels: Optional[Tensor] = None,
        complete_labels: Optional[Tensor] = None,
    ) -> Tuple[Optional[Tensor], Tensor, Optional[Tensor]]:

        # pass through BERT to get contextualized embeddings
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=self.use_last_n_layers > 1)

        bert_out = outputs.last_hidden_state
        if self.use_last_n_layers > 1:
            # for each token, concatenate the hidden states of the last n layers
            hs_to_use = outputs.hidden_states[-self.use_last_n_layers:]
            if self.bert_reduction == 'cat':
                bert_out = torch.cat(hs_to_use, dim=2)
            elif self.bert_reduction == 'mean':
                bert_out = torch.stack(hs_to_use, dim=0).mean(dim=0)
            elif self.bert_reduction == 'sum':
                bert_out = torch.stack(hs_to_use, dim=0).sum(dim=0)

        bert_out = self.features_dropout(bert_out)

        # pass through LSTM
        features_out = bert_out
        if self.use_lstm:
            features_out, (_, _) = self.lstm(features_out)
        features_out = self.features_dropout(features_out)

        # extract for each sentence the position of its B-coref
        logits = self.classifier(features_out.flatten(start_dim=1, end_dim=2))

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        new_logits = logits.detach().clone()
        # logic to output the best prediction for each sentence
        with torch.no_grad():
            batch_size = input_ids.shape[0]

            # compute softmax to transform logits into probabilities
            probabilities = functional.softmax(new_logits, dim=-1)

            entities = self.nertagger(input_ids,
                                      attention_mask).logits.argmax(dim=-1)

            # this ner tagger doesn't seem to use B-tags properly, so fix that
            for s in range(batch_size):
                for w in range(input_ids.shape[1]):
                    tag = entities[s, w]
                    if tag == 0:
                        continue
                    # if a I-x is not after a B-x, change it to B-x
                    if tag % 2 == 0 and (w == 0 or
                                         (entities[s, w - 1] != tag - 1) and
                                         entities[s, w - 1] != tag):
                        entities[s, w] = tag - 1

            # force "nothing" tag on paddings just to be safe
            entities[attention_mask == 0] = 0

            # check that the correct coreference is actually one of the entities
            if complete_labels is not None:
                self.entities_total += len(complete_labels)
                for s in range(len(complete_labels)):
                    # count the cases in which the pronoun is before the entity
                    # if ((complete_labels[s] == 2).sum().item() == 0 or
                    #     (complete_labels[s] == 1).nonzero().item() <
                    #     (complete_labels[s] == 2).nonzero().item()):
                    #     self.count_pron_before += 1

                    # if there is no coreference, we consider entities correct
                    if (complete_labels[s] == 2).sum().item() == 0:
                        self.entities_correct += 1
                        continue
                    # the token labeled 2 needs to have a B-tag
                    b_index = (complete_labels[s] == 2).nonzero().item()
                    b_label = entities[s, b_index].item()
                    if b_label % 2 == 0:
                        # in this case b_label is recognized as an I-tag: wrong
                        continue
                    # if this is correct, all the tokens labeled 3 need an I-tag
                    i_tags_indices = (
                        complete_labels[s] == 3).nonzero().flatten()
                    if not (entities[s, i_tags_indices]
                            == b_label + 1).all().item():
                        continue
                    # also, the recognized entity has to stop at the same point
                    last_tag = (i_tags_indices[-1].item()
                                if len(i_tags_indices) else b_index)
                    if entities[s, last_tag + 1] != (b_label + 1):
                        # congrats! the correct coreference was between the
                        # entities recognized
                        self.entities_correct += 1

            # filter out the classes (positions) not recognized as B-per
            choices = entities == 3
            probabilities[:, :-1] = probabilities[:, :-1] * choices

            # between the possible choices, choose the one with highest prob
            b_predictions = probabilities.argmax(dim=-1)

            # now we create the actual prediction vector
            predictions = torch.zeros_like(input_ids)

            # set the prediction for words we know are padding to -1
            predictions[attention_mask == 0] = -1

            # for the pronoun, we have no choices to make
            predictions[torch.arange(batch_size), pron_positions] = 1

            # set B-tags according to our predictions.
            # after, set I-tags according to the nertager
            # predictions[torch.arange(batch_size), b_predictions] = 2
            for s in range(batch_size):
                if b_predictions[s] == self.max_len:
                    continue
                b_position = b_predictions[s]
                predictions[s, b_position] = 2
                j = 1
                while (b_position + j < predictions.shape[1] and
                       (entities[s, b_position + j]
                        == entities[s, b_position] + 1)):

                    predictions[s, b_position + j] = 3
                    j += 1

            # take into account only the entities that appear before the pronoun
            # edit: not really worth
            # for s in range(batch_size):
            #     entities[s, pron_positions[s]:] = 0

            # if they are all negative (b-coref is never the highest confidence
            # # class) there is no b-coref and therefore no i-coref
            # if (distances > 0).sum().item() == 0:
            #     return loss, probabilities, predictions

            # ----------------------- RANDOM -----------------------
            # b_coref_indices = []
            # for s in range(batch_size):
            #     choices = (entities[s] == 3).nonzero().flatten().tolist()
            #     if len(choices) == 0:
            #         chosen = -1
            #     else:
            #         chosen = random.choice(choices)
            #         predictions[s, chosen] = 2
            #     b_coref_indices.append(chosen)
            # self.n_bper_per_sentence.append(len(choices))
            # ----------------------- RANDOM -----------------------

            # b_coref_indices = torch.tensor(b_coref_indices)

            # set to label 2 the max confidence B-tag
            # b_coref_indices = distances.argmax(dim=-1)

            # predictions[torch.arange(predictions.shape[0]), b_coref_indices] = 2

            # if labels is not None:
            #     for s in range(len(labels)):
            #         if predictions[s].equal(labels[s]):
            #             print('found a correct prediction in forward!')

        return loss, logits, predictions

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pron_positions: Tensor,
        a_marker_indices: Optional[Tensor],
        b_marker_indices: Optional[Tensor],
        labels: Optional[Tensor] = None,
    ) -> Tuple[Optional[Tensor], Tensor, Optional[Tensor]]:

        if not self.show_candidates:
            return self.forward_no_candidates(input_ids, attention_mask,
                                              pron_positions, labels)
        else:
            if a_marker_indices is None or b_marker_indices is None:
                raise ValueError('Marker indices for candidates are None')

        # pass through BERT
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=self.use_last_n_layers > 1,
        )

        bert_out = outputs.last_hidden_state
        if self.use_last_n_layers > 1:
            # for each token, concatenate the hidden states of the last n layers
            hs_to_use = outputs.hidden_states[-self.use_last_n_layers:]
            if self.bert_reduction == 'cat':
                bert_out = torch.cat(hs_to_use, dim=2)
            elif self.bert_reduction == 'mean':
                bert_out = torch.stack(hs_to_use, dim=0).mean(dim=0)
            elif self.bert_reduction == 'sum':
                bert_out = torch.stack(hs_to_use, dim=0).sum(dim=0)

        # bert Dropout
        bert_out = self.features_dropout(bert_out)

        features_out = bert_out
        # pass through LSTM
        if self.use_lstm:
            features_out, (_, _) = self.lstm(bert_out)

        # keep only the pronoun token for each sentence
        pron_outputs = features_out[torch.arange(features_out.shape[0]),
                                    pron_positions, :]

        # find for each sentence the indices of the vectors coming from a and b
        a_mask = torch.zeros_like(features_out)
        b_mask = torch.zeros_like(features_out)
        a = a_marker_indices.reshape(-1, 2).t()
        b = b_marker_indices.reshape(-1, 2).t()
        a[0] += 1
        b[0] += 1
        a_mask[torch.arange(a_mask.shape[0]), a] = 1
        b_mask[torch.arange(b_mask.shape[0]), b] = 1
        a_mask = a_mask.cumsum(dim=1)
        b_mask = b_mask.cumsum(dim=1)
        a_mask = a_mask == 1
        b_mask = b_mask == 1

        # for each sentence, how many vectors represent their A/B candidate
        a_lengths = a[1] - a[0]
        b_lengths = b[1] - b[0]

        # zero out every output that is not coming from an A/B candidate
        a_outputs = features_out * a_mask
        b_outputs = features_out * b_mask

        # manually take the mean vector for every A/B candidate
        a_means = a_outputs.sum(dim=1) / a_lengths.unsqueeze(1)
        b_means = b_outputs.sum(dim=1) / b_lengths.unsqueeze(1)
        vectors_to_concat = [pron_outputs, a_means, b_means]

        # find indices of the last real (not padding) token in each sentence
        if self.use_lstm_last_out:
            last_token_indices = attention_mask.min(dim=-1)[1] - 1
            last_token_vectors = features_out[
                torch.arange(features_out.shape[0]), last_token_indices]
            vectors_to_concat.append(last_token_vectors)

        pab_output = torch.cat(vectors_to_concat, dim=1)

        #feature dropout
        if self.features_dropout_p > 0:
            pab_output = self.features_dropout(pab_output)

        # pass through classifier
        logits = self.classifier(pab_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return loss, logits, None

    # def predict_sentence_no_candidates(self, sentence: str):
    #     self.eval()

    #     return

    def predict_sentence_candidates(self, sentence: str):
        self.eval()
        t = self.tokenizer(sentence,
                           add_special_tokens=False,
                           return_tensors='pt')
        p_encoding = self.tokenizer.encode(
            '[P]',
            add_special_tokens=False,
        )[0]
        a_encoding = self.tokenizer.encode(
            '[A]',
            add_special_tokens=False,
        )[0]
        b_encoding = self.tokenizer.encode(
            '[B]',
            add_special_tokens=False,
        )[0]

        p_index = (t.input_ids == p_encoding).nonzero(as_tuple=True)[1][:1] + 1
        a_marker_indices = (t.input_ids == a_encoding).nonzero(as_tuple=True)[1]
        b_marker_indices = (t.input_ids == b_encoding).nonzero(as_tuple=True)[1]

        _, logits, _ = self.forward(t.input_ids, t.attention_mask, p_index,
                                    a_marker_indices, b_marker_indices, None)

        prediction = logits.argmax(dim=-1)

        return prediction.item()

    def step(self, batch, stage: str):
        if self.show_candidates:
            (input_ids, attention_mask, labels, pron_positions,
             a_marker_indices, b_marker_indices) = batch
        else:
            (input_ids, attention_mask, labels, complete_labels, pron_positions,
             a_marker_indices, b_marker_indices) = batch
        # logits = self.forward(input_ids, attention_mask)
        # loss = self.loss_fn(logits, labels)
        loss, logits, predictions = self.forward(input_ids, attention_mask,
                                                 pron_positions,
                                                 a_marker_indices,
                                                 b_marker_indices, labels)
        if not self.show_candidates:
            if predictions is None:
                raise ValueError('No predictions provided')
            if stage.lower() == 'train':
                self.total_train += predictions.shape[0]
            else:
                self.total_valid += predictions.shape[0]

            for i in range(predictions.shape[0]):

                if predictions[i].equal(complete_labels[i]):
                    if stage.lower() == 'train':
                        self.correct_train += 1
                    else:
                        self.correct_valid += 1
                        # print('found a correct prediction!')
        if stage.lower() == 'train':
            acc = self.train_acc
        elif stage.lower() == 'validation':
            acc = self.valid_acc

        elif stage.lower() == 'test':
            acc = self.test_acc
        else:
            raise ValueError('Invalid stage: ' + stage)
        acc(logits, labels)

        self.log(
            f'{stage} Loss',
            loss,  # type: ignore
            on_step=stage.lower() == 'train',
            # on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        self.log(
            f'{stage} Accuracy',
            acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=self.batch_size,
        )

        return loss

    def training_step(self, batch, _):
        loss = self.step(batch, stage='Train')
        return loss

    def test_step(self, batch, _):
        loss = self.step(batch, stage='Test')
        return loss

    def validation_step(self, batch, _):
        loss = self.step(batch, stage='Validation')
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self.show_candidates:
            self.log(
                'Validation Effective Accuracy',
                self.correct_valid / self.total_valid,
                on_epoch=True,
                prog_bar=True,
            )

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end()

    def on_validation_epoch_start(self) -> None:
        self.total_train = self.total_valid = 0
        self.correct_train = self.correct_valid = 0
        self.entities_correct = self.entities_total = 0

    def on_test_epoch_start(self) -> None:
        self.on_validation_epoch_start()

    def configure_optimizers(self):
        # Prepare optimizer
        if self.clf_weight_decay is None:
            param_optimizer = list(self.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [{
                'params': [
                    p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.01
            }, {
                'params': [
                    p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0
            }]
        else:
            base_parameters = ([
                p for n, p in self.named_parameters() if 'classifier' not in n
            ])
            optimizer_grouped_parameters = [
                {
                    'params': base_parameters,
                },
                {
                    'params': self.classifier.parameters(),
                    'weight_decay': self.clf_weight_decay
                },
            ]

        optimizer = BertAdam(self.parameters() if self.uniform_weight_decay else
                             optimizer_grouped_parameters,
                             lr=self.lr,
                             warmup=0.1,
                             weight_decay=0.01)

        return optimizer
        # return torch.optim.Adam(
        #     self.parameters(),
        #     lr=self.lr,
        #     weight_decay=0.01,
        #     eps=1e-6,
        # )


def predict_and_write(model: GapModel,
                      sentences: List[Dict],
                      outfile: Optional[Path] = None) -> List[int]:
    predictions = []
    dataset: GapDataset = GapDataset(sentences, model.tokenizer)
    outstring = ''
    model.eval()
    for sentence, _, sentence_id in tqdm(dataset,
                                         desc='Predicting...',
                                         total=len(dataset)):

        prediction = model.predict_sentence_candidates(sentence)
        predictions.append(prediction)

        a_coref = prediction == 1
        b_coref = prediction == 2

        outstring += f'{sentence_id}\t{a_coref}\t{b_coref}\n'

    if outfile is not None:
        outfile.write_text(outstring)

    return predictions
