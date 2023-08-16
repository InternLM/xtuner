# Copyright (c) OpenMMLab. All rights reserved.
import transformers
from transformers import Trainer
from xtuner.apis.models import internlm_7b_qlora
from xtuner.apis.datasets import alpaca_dataset, alpaca_data_collator
from xtuner.apis.training_args import DefaultTrainingArguments


def train():
    # get DefaultTrainingArguments and to be updated with passed args
    parser = transformers.HfArgumentParser(DefaultTrainingArguments)
    training_args = parser.parse_args_into_dataclasses()[0]

    # init model and dataset
    model, tokenizer = internlm_7b_qlora(model_name_or_path=training_args.model_name_or_path,return_tokenizer=True)
    train_dataset = alpaca_dataset(tokenizer=tokenizer, path=training_args.dataset_name_or_path)
    data_collator = alpaca_data_collator(return_hf_format=True)
    
    # build trainer
    trainer = Trainer(model=model, 
                      args=training_args, 
                      train_dataset=train_dataset,
                      data_collator=data_collator)
    
    # trianing
    trainer.train()
    
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == '__main__':
    train()
