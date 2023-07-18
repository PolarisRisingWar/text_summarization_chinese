本项目致力于整合各文本摘要模型，将其转为中文文本数据可以直接运行的形式。

本Markdown文件只保证VSCode上MPE插件的浏览效果，因此在其他Markdown渲染工具上效果可能较差。

[TOC]

# 1. 项目简介
本项目的两大任务：1. 集成各重要文本摘要模型的中文输入数据解决方案，优先集成已写好的代码，在此基础上用原生PyTorch和一些常用包来集成自己的代码。2. 集成目前网络上公开的中文文本摘要数据集，提供预处理的工具。  
（另外我还准备用PyTorch统一集成各种摘要生成模型，但是这个flag比较大，以后再拔吧）  
本项目是由于作者太菜，一直困于如何将原本在英文文本数据上运行的各文本摘要模型转换为中文可以运行的格式。所以我觉得大概别人也会有这种困惑，所以决定展开这个项目，以一边激励自己解决这一问题，一边帮助他人解决这一问题。  
所以这个项目就会总结各种重要的文本摘要模型，尤其是提出其在中文文本数据上直接运行的方案。  
有一些代码是其他人写的汉化版，我可能有所修改。具体内容请看具体模型。
# 2. 文本摘要模型简介
（不一定都是专门做文本摘要的，可能有一些泛Text2Text的工作我也姑且放在这里）

本项目所实现的模型：
- [LEAD-3](./models/lead3)
- [TextRank](./models/textrank)
- [TransformerSum - extractive](./models/transformersum/extractive)
- [transformers - pipeline](./models/transformers_pipeline)
- [CPT](./models/cpt)

各模型详细介绍见指定文件夹。
# 3. 数据操作简介
（在2023.5.10修改过一次，所以在此之前的模型可能不可用）
本项目需要的原始数据格式为：  
（暂时不支持非UTF-8编码格式的文本）  
所有数据按照训练集（train）、验证集（valid）、测试集（test）进行划分，3个文件都以jsonl为后缀，每一行是一个用`json.dumps()`储存的JSON对象，元素依次是src（未分割的原文）、tokenized_src（经过分词/分句后的原文）、tgt（未分割的摘要）、tokenized_tgt（经过分词/分句后的摘要）  
分词/句后的对象：①不存在该键 ②仅分词/仅分句→字符串列表 ③分词且分句→nested list，词组成句，句组成文档  

使用文件夹路径来调取数据

（以下内容还没更新）
除label文件（抽取式摘要标签）外的数据示例文件已在[datasets_example](./datasets_example)中给出，提供每个数据集各划分的前100条样本。

其他辅助功能：
1. `text_summarization_chinese/preproess_data/calculate_average_train_tgt_sentences_num.py`：计算`train.tgt`的摘要平均句长，以作为抽取式摘要输出句长的参考
# 4. 数据集简介
本项目提供对一些公开数据集的数据预处理脚本。
对数据集的介绍可参考我写的博文：[文本摘要数据集的整理、总结及介绍（持续更新ing...）_诸神缄默不语的博客-CSDN博客](https://blog.csdn.net/PolarisRisingWar/article/details/122987556)

本项目所提供的数据集：
- [LCSTS数据集](./preproess_data/specific_datasets/lcsts.py)（暂时还未支持生成式语料转抽取式语料的功能）
# 4. 结果评估
未来计划增加使用rouge等评估指标来评估运行结果的功能。目前只有输出结果。
# 5. 工作计划
增加模型/工具包：
- [x] 无监督抽取式摘要：LEAD-3
- [x] 无监督抽取式摘要：TextRank
- [ ] 有监督抽取式摘要：BertSum
- [ ] 有监督抽取式摘要：PreSumm
- [ ] 有监督抽取式摘要：MatchSum
- [ ] 有监督抽取式摘要：REFRESH
- [ ] 有监督抽取式摘要：NeuSum
- [ ] 有监督抽取式摘要：BanditSum
- [ ] 有监督生成式摘要：Pointer-Generator Network
- [ ] 有监督生成式摘要：UniLM
- [x] 有监督生成式摘要：CPT
- [x] 工具包：transformers
- [ ] 工具包：TransformerSum
- [ ] 工具包：bert4keras
- [ ] 工具包：bert_seq2seq
- [ ] 在各模型的README文件内介绍模型原理
- [ ] 增加对单机单卡/多卡运行的配置

数据预处理：
- [ ] 改进原始数据格式（增加对数据过大造成需要拆分到多个文件中、文本含有换行符、空格等特殊符号的情况的支持）
- [ ] 增加生成式语料转抽取式语料的功能
- [x] 增加原始数据示例样本
- [ ] 增加模型运行结果示例样本
- [ ] 增加自动分词和分句功能
- [ ] 增加preprocess_data文件夹的README文件
- [ ] 增加使用抽取式摘要标签数量来计算抽取式摘要任务应该抽取多少句话的参考值的代码
- [ ] 补LCSTS模型抽取式摘要的标签

模型评估：
- [ ] 增加rouge和分类的评估方式
- [ ] 增加各数据集的结果评估比较

其他：
- [x] 根据GitHub对Markdown的支持，优化文档排版
- [ ] 模型的各种可复现性相关的问题

# 6. 本项目所使用的、未在各模型独立文件夹中提及的其他参考资料
1. 抽取式模型整合网站：[Extractive Text Summarization | Papers With Code](https://paperswithcode.com/task/extractive-document-summarization)
2. 代码中使用到zipfile包的部分参考了如下资料：[3.7Python之解压缩ZIP文件_阿兵-AI医疗的专栏-CSDN博客_python zip解压](https://blog.csdn.net/webzhuce/article/details/79950027)
3. 代码中使用到os包的部分参考了如下资料：[Python - 调用终端执行命令_Noah1995的博客-CSDN博客_python执行终端命令](https://blog.csdn.net/weixin_42368421/article/details/98625365)；[python3 创建文件夹_python3.2对文件和文件夹操作_weixin_39672680的博客-CSDN博客](https://blog.csdn.net/weixin_39672680/article/details/110284950)；[python3遍历目录查找文件_昆兰.沃斯 的博客-CSDN博客_python3遍历目录查找文件](https://blog.csdn.net/qq_36523839/article/details/72974517)
4. 代码中使用到raise语句参考了如下资料：[Python3 错误和异常 | 菜鸟教程](https://www.runoob.com/python3/python3-errors-execptions.html)
5. <https://blog.nghuyong.top/2021/05/24/NLP/chinese-pretrained-seq2seq/>：还准备把这里面的各种模型继续集成到本项目中