# transformers.Pipeline

使用huggingface开发的transformers包（文档：[🤗 Transformers](https://huggingface.co/docs/transformers/index)）的Pipeline类直接进行summarization任务的推理。
对Pipeline的简单介绍可以参考我撰写的博文：[huggingface.transformers速成笔记_诸神缄默不语的博客-CSDN博客](https://blog.csdn.net/PolarisRisingWar/article/details/123169526)第一节。

作者在测试时使用的transformers版本为4.12.5，sentencepiece版本为0.1.96（如果没有安装这个包会报错。参考：[python - Transformers v4.x: Convert slow tokenizer to fast tokenizer - Stack Overflow](https://stackoverflow.com/questions/65431837/transformers-v4-x-convert-slow-tokenizer-to-fast-tokenizer)）。
由于pipeline可直接调用的模型列表（https://huggingface.co/models?filter=summarization）中没有专门针对中文语料预训练的模型，所以我只能选择了其中也许是唯一的多语言模型[csebuetnlp/mT5_multilingual_XLSum](https://huggingface.co/csebuetnlp/mT5_multilingual_XLSum)。事实上其他能直接调用SummarizationPipeline的预训练模型也可以，但是这不是都不支持中文嘛就没整。

直接调用`main.py`即可。
入参：
`--data_file_path`&emsp;&emsp;用于生成摘要的src文件路径  
`--result_file_path`&emsp;&emsp;输出结果地址（是一个可以直接通过`w`模式的`open()`函数打开的文本文件），每一行是一条摘要生成结果。  
`--pretrained_model_or_path`&emsp;&emsp;预训练模型或地址。默认值即是`csebuetnlp/mT5_multilingual_XLSum`。改用本地路径的话就需要将预训练模型的这些文件下到本地：  
[![b87cnS.png](https://s4.ax1x.com/2022/03/02/b87cnS.png)](https://imgtu.com/i/b87cnS)  
然后将文件夹路径作为参数传入。
如果以后有其他支持SummarizationPipeline的中文预训练模型，做法也差不多。但是目前还没有，所以我就不考虑了。
这个模型的主要问题在于它的预训练数据都是新闻文本，所以生成的摘要特别新闻风……我这样干说可能不太好理解，等我拿个公开数据集跑一下示例结果就好理解了。

暂时不支持文本中自带`</sentence>`字样的情况。

工作计划：
- [ ] 对文本自带`</sentence>`字样情况的支持
- [ ] 增加对其他pipeline入参调整的支持（如num_workers, max_length等）