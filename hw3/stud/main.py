#%% imports
import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import WandbLogger
import transformers
import logging
import wandb
import config
import torch
from gap_model import GapModel
from gap_data import GapDatamodule
import gap_model, gap_data
from pathlib import Path


def write_predictions(checkpoint_name: str):
    sentences = gap_data.read_dataset(str(config.DEV))
    m = GapModel.load_from_checkpoint(
        str(config.MODEL / f'{checkpoint_name}.ckpt'))

    gap_model.predict_and_write(m, sentences, config.DATA / 'predictions.tsv')


def test_model(checkpoint_name: str):
    test_m = GapModel.load_from_checkpoint(
        str(config.MODEL / f'{checkpoint_name}.ckpt'))
    test_d = GapDatamodule(
        data_dir=config.DATA,
        tokenizer=test_m.tokenizer,
        batch_size=test_m.batch_size,
        upsample_neither=False,
    )
    test_d.setup()
    test_trainer = pl.Trainer(accelerator='gpu',)
    return test_trainer.test(test_m, dataloaders=d.val_dataloader())


#%% seed everything, set up loggers and checkpoints
pl.seed_everything(2048)

transformers.logging.set_verbosity_error()
logging.getLogger('pytorch_lightning').setLevel(logging.INFO)
logger = WandbLogger(
    name='no-candidates',
    project='gap-coreference',
    entity='thekoln',
)

# wandb.define_metric('Validation Accuracy', step_metric='epoch')
wandb.define_metric('Validation Effective Accuracy', step_metric='epoch')

checkpoint_callback = callbacks.ModelCheckpoint(
    config.CKP, monitor='Validation Effective Accuracy', mode='max')

#%% instantiate model, datamodule and trainer
model = GapModel(
    lr=4e-6,
    batch_size=4,
    freeze_bert=False,
    ckp_id='distilbert-base-cased',
    use_last_n_layers=1,
    double_linear=False,
    clf_hidden_dim=1024,
    clf_dropout=0.5,
    bert_dropout=0.1,
    features_dropout=0.5,
    bert_reduction='sum',
    uniform_weight_decay=False,
    use_lstm=False,
    use_lstm_last_out=False,
    lstm_hidden_dim=512,
    lstm_n_layers=1,
    label_smoothing=0.0,
    clf_weight_decay=None,
    show_candidates=False,
    max_len=400,
)
# model = GapModel.load_from_checkpoint(str(config.MODEL / 'distilbert23.ckpt'))

bert_tokenizer = model.tokenizer
d = GapDatamodule(
    data_dir=config.DATA,
    tokenizer=bert_tokenizer,
    batch_size=model.batch_size,
    upsample_neither=True,
    show_candidates=model.show_candidates,
    max_len=400,
)
d.setup()

trainer = pl.Trainer(
    max_epochs=20,
    accelerator='gpu',
    logger=logger,
    callbacks=[checkpoint_callback],
)

# s, m, l = next(iter(d.val_dataloader()))

#%% train, test
# trainer.test(model, dataloaders=d.val_dataloader())
trainer.fit(model, d)
trainer.test(model, dataloaders=d.val_dataloader())
wandb.finish()
