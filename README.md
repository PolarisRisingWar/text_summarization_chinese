本项目致力于整合各文本摘要模型，将其转为中文文本数据可以直接运行的形式。

本Markdown文件只保证VSCode上MPE插件的浏览效果，因此在其他Markdown渲染工具上效果可能较差。

[TOC]

# 1. 项目简介
本项目是由于作者太菜，一直困于如何将原本在英文文本数据上运行的各文本摘要模型转换为中文可以运行的格式。所以我觉得大概别人也会有这种困惑，所以决定展开这个项目，以一边激励自己解决这一问题，一边帮助他人解决这一问题。
所以这个项目就会总结各种重要的文本摘要模型，尤其是提出其在中文文本数据上直接运行的方案。
有一些代码是其他人写的汉化版，我可能有所修改。具体内容请看具体模型。
# 2. 文本摘要模型简介
本项目所实现的模型：
- [LEAD-3](./models/lead3)
- [TextRank](./models/textrank)
- [TransformerSum - extractive](./models/transformersum/extractive)

本项目所使用的各种模型的介绍，都会逐渐更新在我的CSDN博文上：[文本摘要（text summarization）任务的一般研究范式及相对应的重要模型简介（持续更新ing...）_诸神缄默不语的博客-CSDN博客](https://blog.csdn.net/PolarisRisingWar/article/details/123090751)
# 3. 数据操作简介
本项目需要的原始数据格式为：
（暂时不支持含换行符、非UTF-8编码格式的文本）
所有数据按照训练集（train）、验证集（val）、测试集（test）进行划分，如果提前已经做好分词在数据集划分名称后加`_tok`。
每条数据一行，原文`.src`后缀，含摘要`.tgt`后缀。分词好的格式为以空格隔开每个token文本（因此不支持含空格的文本）。
抽取式标签后缀`.label`，仅支持通过`json.dump`保存的list。如果提前做好分句，可以在原文中用`</sentence>`token作为标识。在抽取式摘要中会使用该标识，在生成式摘要中会自动去掉该标识（以后会提供支持原文中自带`</sentence>`文本的功能）。

使用文件夹名（数据集名）来调取数据。文件夹下的文件为：
[![blnM26.png](https://s4.ax1x.com/2022/03/01/blnM26.png)](https://imgtu.com/i/blnM26)

示例数据稍后将会在text_summarization_chinese/unified_data文件夹中给出。

其他辅助功能：
1. `text_summarization_chinese/preproess_data/calculate_average_train_tgt_sentences_num.py`：计算`train.tgt`的摘要平均句长，以作为抽取式摘要输出句长的参考
# 4. 结果评估
未来计划增加使用rouge等评估指标来评估运行结果的功能。目前只有输出结果。
# 5. 工作计划及项目日志
增加模型：
- [x] 无监督抽取式摘要：LEAD-3
- [x] 无监督抽取式摘要：TextRank
- [ ] 有监督抽取式摘要：BertSum
- [ ] 有监督抽取式摘要：PreSumm
- [ ] 有监督抽取式摘要：MatchSum
- [ ] 有监督抽取式摘要：REFRESH
- [ ] 有监督抽取式摘要：NeuSum
- [ ] 有监督抽取式摘要：BanditSum
- [x] 有监督抽取式摘要：TransformerSum-extractive
- [ ] 有监督生成式摘要：Pointer-Generator Network
- [ ] 有监督生成式摘要：TransformerSum-abstractive
- [ ] 有监督生成式摘要：transformers-pipeline（仅支持推理）

数据预处理：
- [ ] 改进原始数据格式（增加对数据过大造成需要拆分到多个文件中、文本含有换行符等特殊符号的情况的支持）
- [ ] 增加生成式语料转抽取式语料的功能
- [ ] 增加示例样本
- [ ] 增加自动分词和分句功能
- [ ] 增加preprocess_data文件夹的README文件

模型评估：
- [ ] 增加rouge和分类的评估方式

# 6. 本项目所使用的、未在各模型独立文件夹中提及的其他参考资料
1. 抽取式模型整合网站：[Extractive Text Summarization | Papers With Code](https://paperswithcode.com/task/extractive-document-summarization)