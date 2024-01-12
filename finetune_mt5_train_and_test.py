import random
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import (
    AdamW,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    get_linear_schedule_with_warmup
)

from datasetHelper import get_dataset

from tqdm.notebook import tqdm
import copy

import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from lightning.pytorch.loggers import WandbLogger
import wandb

import torchmetrics
from torchmetrics.text import BLEUScore
from torchmetrics.text.bert import BERTScore
from torchmetrics.text.rouge import ROUGEScore

from sklearn.metrics import f1_score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score


parser = argparse.ArgumentParser(description='Add args through command line')
parser.add_argument('--dataset', type=str, default="med_qa", help='dataset name')
parser.add_argument('--batch_size', type=int, default=12, help='batch size')
parser.add_argument('--run_name', type=str, default="run_1", help='run name')
parser.add_argument('--load_from_checkpoint', type=bool, default=False, help='whether load from checkpoint')
parser.add_argument('--load_path', type=str, default=None, help='if load from checkpoint, specify load path')
parser.add_argument('--mode', type=str, default="train", help='train or test')
parser.add_argument('--project_name', type=str, default="mt5-med_qa", help='project name')
args = parser.parse_args()


class Finetune_mT5_Dataset(Dataset):
    def __init__(self, tokenizer, data_list, max_len_inp=256,max_len_out=256):

        self.data_tuples = data_list

        self.max_len_input = max_len_inp
        self.max_len_output = max_len_out
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.skippedcount =0
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        labels = copy.deepcopy(target_ids)
        labels [labels==0] = -100

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask,"labels":labels}

    def _build(self):
        for inputs,outputs in self.data_tuples:
          input_sent = "Question: "+inputs
          ouput_sent = "Answer: "+outputs

          # tokenize inputs
          tokenized_inputs = self.tokenizer.batch_encode_plus(
              [input_sent], max_length=self.max_len_input, padding="max_length", return_tensors="pt", truncation=True
          )
          # tokenize targets
          tokenized_targets = self.tokenizer.batch_encode_plus(
              [ouput_sent], max_length=self.max_len_output, padding="max_length",return_tensors="pt", truncation=True
          )

          self.inputs.append(tokenized_inputs)
          self.targets.append(tokenized_targets)

class mT5FineTuner(pl.LightningModule):
    def __init__(self,hparams, mt5model, mt5tokenizer, train_dataset, validation_dataset, test_dataset):
        super(mT5FineTuner, self).__init__()
        #self.hparams = hparams
        self.model = mt5model
        self.tokenizer = mt5tokenizer
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

    def forward( self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids = batch["target_ids"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )
        loss = outputs[0]
        self.log('train_loss',loss)
        return loss

    def validation_step(self, batch, batch_idx):

        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids=batch["target_ids"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )
        loss = outputs[0]

        generated_ids = self.model.generate(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            max_length=50,
            num_beams=5,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )
        generated_sentences = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
        original_sentences = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in batch["target_ids"]]
        smooth = SmoothingFunction()
        bleu_score = corpus_bleu([[ref.split()] for ref in original_sentences], [gen.split() for gen in generated_sentences], weights=(0.5, 0.5), smoothing_function=smooth.method1)

        rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores_list = [rouge_scorer_instance.score(gen_sent, orig_sent) for gen_sent, orig_sent in zip(generated_sentences, original_sentences)]

        avg_rouge1 = sum([score['rouge1'].fmeasure for score in rouge_scores_list]) / len(rouge_scores_list)
        avg_rouge2 = sum([score['rouge2'].fmeasure for score in rouge_scores_list]) / len(rouge_scores_list)
        avg_rougeL = sum([score['rougeL'].fmeasure for score in rouge_scores_list]) / len(rouge_scores_list)

        self.log("val_loss", loss)
        self.log("bleu_score", bleu_score, prog_bar=True)
        self.log("avg_rouge1", avg_rouge1, prog_bar=True)
        self.log("avg_rouge2", avg_rouge2, prog_bar=True)
        self.log("avg_rougeL", avg_rougeL, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids = batch["target_ids"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )
        loss = outputs[0]
        self.log("test_loss",loss)
        return loss

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=args.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=args.batch_size,num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=3e-4, eps=1e-8)
        return optimizer


def train(model):
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='./checkpoints/mt5-small-dataset-{epoch:02d}-{val_loss:.2f}'+f"-{args.dataset}-{args.run_name}",
        save_top_k=2,
        mode='min',
        save_last=True
    )

    #for debug
    #trainer = pl.Trainer(max_epochs = 2,callbacks=[checkpoint_callback],limit_train_batches=0.1,logger=wandb_logger,val_check_interval=0.5)
    #end debug
    trainer = pl.Trainer(max_epochs = 2,callbacks=[checkpoint_callback],logger=wandb_logger,val_check_interval=0.1)

    trainer.fit(model)

    trainer.test(dataloaders = model.test_dataloader())


if __name__ == "__main__":
    #init:
    pl.seed_everything(42)

    #model and tokenizer
    mt5_model = transformers.MT5ForConditionalGeneration.from_pretrained(
            "/ceph/home/liuxingyu/NLP/final/models/mt5-small-new"
        )
    mt5_tokenizer = transformers.MT5Tokenizer.from_pretrained(
            "/ceph/home/liuxingyu/NLP/final/models/mt5-small-new"
        )

    # dataset preparation
    raw_datasets = get_dataset(args.dataset, " ")
    data_tuples_train = raw_datasets["train"]["tuples"]
    data_tuples_validation = raw_datasets["validation"]["tuples"]
    data_tuples_test = raw_datasets["test"]["tuples"]

    train_dataset = Finetune_mT5_Dataset(mt5_tokenizer,data_tuples_train)
    validation_dataset = Finetune_mT5_Dataset(mt5_tokenizer,data_tuples_validation)
    test_dataset = Finetune_mT5_Dataset(mt5_tokenizer,data_tuples_test)

    if args.load_from_checkpoint:
        #load from checkpoint
        print(f"load model from checkpoint {args.load_path}")
        assert args.load_path is not None
        model = mT5FineTuner.load_from_checkpoint(args.load_path, 
        hparams=args, 
        mt5model=mt5_model, 
        mt5tokenizer=mt5_tokenizer, 
        train_dataset=train_dataset, 
        validation_dataset=validation_dataset, 
        test_dataset=test_dataset
        )
    else:
        model = mT5FineTuner(args,mt5_model,mt5_tokenizer,train_dataset,validation_dataset,test_dataset)

    if args.mode == "train":
        #define logger
        wandb.init(project=f"mt5-{args.project_name}")
        wandb.watch(model)
        wandb.run.name = f"mt5-{args.dataset}-run-{args.run_name}"
        wandb_logger = WandbLogger(log='all')
        
        #train
        train(model)
    elif args.mode == "test":
        print("test mode")
        test_sent = 'Question: A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take? A:Disclose the error to the patient but leave it out of the operative report B:Disclose the error to the patient and put it in the operative report C:Tell the attending that he cannot fail to disclose this mistake D:Report the physician to the ethics committee E:Refuse to dictate the operative report.'
        test_tokenized = mt5_tokenizer.encode_plus(test_sent, return_tensors="pt")

        test_input_ids  = test_tokenized["input_ids"]
        test_attention_mask = test_tokenized["attention_mask"]

        test_tokenized = test_tokenized.to('cuda')
        test_input_ids = test_input_ids.to('cuda')
        test_attention_mask = test_attention_mask.to('cuda')


        model.model.eval()
        beam_outputs = model.model.generate(
            input_ids=test_input_ids,attention_mask=test_attention_mask,
            max_length=64,
            early_stopping=True,
            num_beams=10,
            num_return_sequences=3,
            no_repeat_ngram_size=2
        )

        for beam_output in beam_outputs:
            sent = mt5_tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
            print (sent)