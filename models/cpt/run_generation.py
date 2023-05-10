#代码参考了https://github.com/fastnlp/CPT/blob/master/finetune/generation/run_gen.py
#默认参数参考了https://github.com/fastnlp/CPT/blob/master/finetune/generation/utils.py

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_folder_path",default='datasets_example/lcsts_example',type=str)  #数据储存路径
parser.add_argument("--result_folder_path",default='/data/cpt_output',type=str)  #储存结果的文件夹

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
extracted_result=open('').readlines()

tuiliguocheng_dict=eval(open('').read())

total_data=[json.loads(x) for x in extracted_result]
data_original=[]
data_destination=[]
for d in total_data:
    if d['money']['type']=='qita':
        data_original.append('诉讼请求：'+d['ask']+'  事实描述文本：'+' '.join(d['money']['rationale']))
        susong_id=d['susong_id']
        data_destination.append('推理过程：'+tuiliguocheng_dict[susong_id]+' 最终结果：'+str(d['money']['amount']))

datasets_indexes=random_split_dataset(list(range(len(data_original))),split_ratio,random_seed)

def convert2Dataset(indexes):
    results={'summarization':[],'article':[]}
    for sample_index in indexes:
        results['summarization'].append(data_destination[sample_index])
        results['article'].append(data_original[sample_index])
    return Dataset.from_dict(results)

datasets={}
datasets['train']=convert2Dataset(datasets_indexes[0])
datasets['validation']=convert2Dataset(datasets_indexes[1])
datasets['test']=convert2Dataset(datasets_indexes[2])

# Part 2: 其他设置
outdir=''

# Part 3: 建模
model_path=""

tokenizer=BertTokenizer.from_pretrained(model_path)
model=CPTForConditionalGeneration.from_pretrained(model_path)
target_length=580
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
if "train" not in datasets:
    raise ValueError("--do_train requires a train dataset")
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

    # # rougeLSum expects newline after each sentence
    # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    # labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    while '' in preds:
        idx=preds.index('')
        preds[idx]='。'

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
   
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    for (x,y) in zip(decoded_preds, decoded_labels):
        print('样本')
        print(x)
        print(y)
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
    '--max_source_length='+str(source_length),  #我记得这个是1024-11来着？
    '--val_max_target_length='+str(target_length),
    '--predict_with_generate=1',
    '--seed',str(20200508),
    '--num_train_epochs','8',
    '--save_strategy','no',
    '--evaluation_strategy','epoch',
    '--learning_rate',str(2e-5),
]
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses(args)

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
if training_args.do_train:
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

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