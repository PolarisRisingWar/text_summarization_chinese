# TransformerSum - extractive

首先调用`convert_data_format.py`将本项目采用的原始数据转换为TransformerSum可用的格式（这里有两条注意事项：1. 由于测试部分不知道为什么会出现bug，有些batch里面没有target键，所以测试部分实际上不可用。因此测试部分的代码需要另外跑，不能直接用TransformerSum的`main.py`的代码。跑测试部分的代码见后文。2. 我在代码中分词直接以char为单位进行分割了，我没看懂TransformerSum自带的tokenizer部分代码，如果是直接以这个分词单位进行tokenize的话其实是不合适的，具体为什么不合适此处略，总之不合适。但是也不算大错。总之不合适。transformers包是支持直接输入整句文本的，感觉用这个更合适些，以后我自己改写代码了就改成用直接输入的方式。）。
`convert_data_format.py`入参：
`--dataset_folder_path`&emsp;&emsp;含有train/val/test src/tgt/label文件的文件夹路径。
`--output_folder_path`&emsp;&emsp;处理后的数据输出文件夹。将在该文件夹下储存dataset_name.train/val/test.0.json（3个文件）。
`--dataset_name`&emsp;&emsp;数据输出文件使用的名称。

然后调用TransformerSum的`main.py`代码进行训练操作，可参考的运行代码为：
```
python main.py --data_path dataset_folder_path --weights_save_path mypath/weights --do_train --max_steps 50000 --model_name_or_path mypath/longformer_zh --data_type txt
```
对参数的介绍：
`data_path`可以直接使用`dataset_folder_path`参数。

`weights_save_path`是checkpoint存储的位置。

`model_name_or_path`可以使用任一transformers包支持的autoencoding的预训练模型，对transformers包的使用介绍可以参考我写的博文[huggingface transformers包 文档学习笔记（持续更新ing...）_诸神缄默不语的博客-CSDN博客](https://blog.csdn.net/PolarisRisingWar/article/details/122953984)。注意，这里我用的是[ValkyriaLenneth/Longformer_ZH](https://github.com/ValkyriaLenneth/Longformer_ZH)，它可以直接用LongformerModel来调用模型，这个TransformerSum是原生支持的；但是它的tokenizer需要调用BertTokenizer，不能直接使用TransformerSum中用的AutoTokenizer，因此我需要在TransformerSum的`extractive.py`中import transformers的BertTokenizer，并将原本的第208行`self.tokenizer = AutoTokenizer.from_pretrained(`换成`self.tokenizer = BertTokenizer.from_pretrained(`。

`data_type`在第一次运行时必须要加，这是用来对原始JSON文件（经`convert_data_format.py`处理得到的结果）进行处理的参数。如果选择`txt`，就在`data_path`底下再添加三个同名的txt文件）

其他参数请参考TransformerSum的文档（[Training an Extractive Summarization Model — TransformerSum 1.0.0 documentation](https://transformersum.readthedocs.io/en/latest/extractive/training.html)）

运行结束后在`weights_save_path`下可以找到checkpoint，在TransformerSum的src文件夹下新建Python文件，运行测试代码。（需要注意两点，1. 需要将TransformerSum的`extractive.py`的第1095行`" ".join([token.text for token in sentence if str(token) != "."]) + "."`改成`" ".join([token for token in sentence])`。2. 原始TransformerSum代码没有限制摘要句子按顺序输出，这一点我已经给作者提了issue：[Suggest about the index order of extractive results · Issue #68 · HHousen/TransformerSum](https://github.com/HHousen/TransformerSum/issues/68)，如果作者听的话后期可能会直接改，如果没改的话直接参考我在issue里提出的解决方式即可）
可以参考的写法为：
```python
import argparse
from tqdm import tqdm
from extractive import ExtractiveSummarizer

parser = argparse.ArgumentParser()
parser.add_argument("--file_path",default='dataset_path/test.src',type=str)
parser.add_argument("--result_path",default='results_path/result.txt',type=str)
parser.add_argument('--checkpoint_path',default="mypath/weights/None/some_random_id/checkpoints/epoch=99-step=20299.ckpt",type=str)
parser.add_argument('--num_summary_sentences',default=3,type=int)

args = parser.parse_args()
arg_dict=args.__dict__

model = ExtractiveSummarizer.load_from_checkpoint(arg_dict['checkpoint_path'])

texts=open(arg_dict['file_path']).readlines()
result_file=open(arg_dict['result_path'],'w')
for t in tqdm(texts):
    text_to_summarize=t[:-1]
    predicted_summary=model.predict_sentences(input_sentences=text_to_summarize.\
                    split('</sentence>')[:-1],
                    num_summary_sentences=arg_dict['num_summary_sentences'],tokenized=True)
    result_file.write(predicted_summary.replace(' ','')+'\n')
result_file.close()
```
对参数的介绍：
`--file_path`&emsp;&emsp;需要被摘要的test.src文件路径。
`--result_path`&emsp;&emsp;预测摘要输出路径（是一个需要直接可写入的文本文件）
`--checkpoint_path`&emsp;&emsp;checkpoint路径。
`--num_summary_sentences`&emsp;&emsp;测试输出句长，可以通过`text_summarization_chinese/preproess_data/calculate_average_train_tgt_sentences_num.py`计算训练集摘要平均句长来作为选值参考

工作计划和项目日志：
- [ ] 增加对`dataset_folder_path`和`output_folder_path`使用不严格文件夹路径写法的容错支持
- [ ] 增加对不同模型和tokenizer的支持（如本README文件中提及的longformer_zh模型的问题）
- [ ] 增加自动选择输出句长的功能