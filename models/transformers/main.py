#代码参考了https://huggingface.co/docs/transformers/tasks/summarization

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_folder_path",default='datasets_example/lcsts_example',type=str)
#数据储存路径，即底下有3个jsonl文件的文件夹

parser.add_argument("--pretrained_model_path",default='datasets_example/lcsts_example',type=str)
#预训练模型的名称或路径

parser.add_argument("--result_folder_path",default='/data/transformers_output',type=str)  #储存结果的文件夹

parser.add_argument("--prompt",default='摘要：',type=str)
parser.add_argument("--source_max_length",default=1024,type=int)
parser.add_argument("--target_max_length",default=128,type=int)

args = parser.parse_args()
arg_dict=args.__dict__

import json,os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,DataCollatorForSeq2Seq,Seq2SeqTrainingArguments,TrainerCallback,Seq2SeqTrainer,\
                        HfArgumentParser
import evaluate

from datasets import Dataset,DatasetDict

model_path=arg_dict['pretrained_model_path']

# Part 1: 导入数据集
def convert2Dataset(split_name):
    """将数据集转换为datasets.Dataset的格式"""
    results={'summary':[],'text':[]}
    original_data=open(os.path.join('/data/wanghuijuan/other_data/tsc_data/lcsts',split_name+'.jsonl')).readlines()[:100]
    for sample in original_data:
        sample_jsoned=json.loads(sample)
        results['summary'].append(sample_jsoned['tgt'].strip())
        results['text'].append(sample_jsoned['src'].strip())
    return Dataset.from_dict(results)

datasets={}
datasets['train']=convert2Dataset('train')
datasets['validation']=convert2Dataset('valid')
datasets['test']=convert2Dataset('test')
datasets=DatasetDict(datasets)

# Part 2: 构建模型
tokenizer=AutoTokenizer.from_pretrained(model_path)
model=AutoModelForSeq2SeqLM.from_pretrained(model_path)

def preprocess_function(examples):
    inputs = [arg_dict['prompt'] + doc for doc in examples['text']]
    model_inputs = tokenizer(inputs, max_length=arg_dict['source_max_length'], truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=arg_dict['target_max_length'], truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = datasets.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Part 3: 构建评估机制
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    print(decoded_preds)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

# Part 4：训练
training_args = Seq2SeqTrainingArguments(
    output_dir=arg_dict['result_folder_path'],
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# Part 5：测试
#TODO: 这一部分还没有debug，懒得改了先上传一下
predictions, labels, metrics = trainer.predict(datasets['test'], metric_key_prefix="test")
test_preds = tokenizer.batch_decode(
    predictions, skip_special_tokens=True,
)
test_preds = [pred.strip() for pred in test_preds]
output_test_preds_file = os.path.join(training_args.output_dir, "test_generations.txt")
with open(output_test_preds_file, "w",encoding='UTF-8') as writer:
    writer.write("\n".join(test_preds))