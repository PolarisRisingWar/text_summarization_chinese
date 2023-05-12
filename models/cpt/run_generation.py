#代码参考了https://github.com/fastnlp/CPT/blob/master/finetune/generation/run_gen.py
#默认参数参考了https://github.com/fastnlp/CPT/blob/master/finetune/generation/utils.py

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_folder_path",default='/data/wanghuijuan/other_data/tsc_data/lcsts',type=str)
#数据储存路径

parser.add_argument("--result_folder_path",default='/data/wanghuijuan/cpt_output/tsc_cpt_output',type=str)
#储存结果的文件夹。要求这个文件夹必须存在（因为还没写不存在setting的代码）

parser.add_argument("--pretrained_model_path",default='/data/wanghuijuan/pretrained_model/cpt-large',type=str)

args = parser.parse_args()
arg_dict=args.__dict__

import json,os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from torch.utils.data import Dataset,DataLoader

from modeling_cpt import CPTForConditionalGeneration

import transformers
from transformers import BertTokenizer,DataCollatorForSeq2Seq,Seq2SeqTrainingArguments,TrainerCallback,Seq2SeqTrainer,HfArgumentParser
from transformers.trainer_utils import is_main_process

from datasets import Dataset

# Part 1: 数据集
def convert2Dataset(split_name):
    """将数据集转换为datasets.Dataset的格式"""
    results={'summarization':[],'article':[]}
    original_data=open(os.path.join(arg_dict['dataset_folder_path'],split_name+'.jsonl')).readlines()[:100]
    for sample in original_data:
        sample_jsoned=json.loads(sample)
        results['summarization'].append(sample_jsoned['tgt'].strip())
        results['article'].append(sample_jsoned['src'].strip())
    return Dataset.from_dict(results)

datasets={}
datasets['train']=convert2Dataset('train')
datasets['validation']=convert2Dataset('valid')
datasets['test']=convert2Dataset('test')

# Part 2: 其他设置
outdir=arg_dict['result_folder_path']
model_path=arg_dict['pretrained_model_path']

# Part 3: 建模
tokenizer=BertTokenizer.from_pretrained(model_path)
model=CPTForConditionalGeneration.from_pretrained(model_path)
target_length=512
source_length=1024
model.config.max_length=target_length

text_column='article'
summary_column='summarization'
column_names = datasets["train"].column_names
max_target_length =target_length
padding=False

def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[summary_column]
    model_inputs = tokenizer(inputs, max_length=source_length, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)


    model_inputs["labels"] = labels["input_ids"]
    return model_inputs




train_dataset = datasets["train"]
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=None,
    remove_columns=column_names,
    load_from_cache_file=True,
)


eval_dataset = datasets["validation"]
eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=None,
    remove_columns=column_names,
    load_from_cache_file=True,
)


test_dataset = datasets["test"]
test_dataset = test_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=None,
    remove_columns=column_names,
    load_from_cache_file=True,
)


max_eval_num=30000
if len(eval_dataset)>max_eval_num:
    eval_dataset=Dataset.from_dict(eval_dataset[:max_eval_num])
print(len(eval_dataset))


# Data collator
label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8,
)



# Metric
from rouge import Rouge 
rouge = Rouge()

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    while '' in preds:
        idx=preds.index('')
        preds[idx]='。'

    return preds, labels

def compute_metrics(eval_preds):
    #注意这里有一点在于，如果预测为空值，是有改进方案的；如果标签为空值没有，如果有标签为空的情况需要另行定义，否则会报ValueError
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
   
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    scores = rouge.get_scores(decoded_preds, decoded_labels,avg=True)
    for key in scores:
        scores[key]=scores[key]['f']*100

    result=scores

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

class TestCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
        metrics['epoch']=state.epoch
        state.log_history.append(metrics)

# Initialize our Trainer

args=[
    '--model_name_or_path',model_path,
    '--do_train','--do_eval','--do_predict',
    '--output_dir',outdir,
    '--per_device_train_batch_size','1',
    '--per_device_eval_batch_size','1',
    '--overwrite_output_dir',
    '--max_source_length='+str(source_length),
    '--val_max_target_length='+str(target_length),
    '--predict_with_generate=1',
    '--seed',str(20200508),
    '--num_train_epochs','8',
    '--save_strategy','no',
    '--evaluation_strategy','epoch',
    '--learning_rate',str(2e-5),
]

training_args = Seq2SeqTrainingArguments(output_dir=outdir,overwrite_output_dir=True,do_train=True,do_eval=True,do_predict=True,
                                         evaluation_strategy="epoch",per_device_train_batch_size=1,per_device_eval_batch_size=1,learning_rate=2e-5,
                                         num_train_epochs=8,save_strategy="no",seed=20200508,predict_with_generate=1)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[TestCallback],
)


# Training
train_result = trainer.train()
trainer.save_model()  # Saves the tokenizer too for easy upload

metrics = train_result.metrics
metrics["train_samples"]=len(train_dataset)

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

if trainer.is_world_process_zero():
    if training_args.predict_with_generate:
        predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
        test_preds = tokenizer.batch_decode(
            predictions, skip_special_tokens=True,
        )
        test_preds = [pred.strip() for pred in test_preds]
        output_test_preds_file = os.path.join(training_args.output_dir, "test_generations.txt")
        with open(output_test_preds_file, "w",encoding='UTF-8') as writer:
            writer.write("\n".join(test_preds))