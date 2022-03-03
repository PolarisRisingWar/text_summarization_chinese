# TextRank

使用textrank4zh包（[letiantian/TextRank4ZH: 从中文文本中自动提取关键词和摘要](https://github.com/letiantian/TextRank4ZH)）实现。  
直接在测试集上运行并得到结果。  
注意textrank4zh包有个问题，它会自己做一个分句，可能在数据集原本的分句方式的基础上二度分句。我在本代码中对此的处理方式就是将安装textrank4zh位置的`util.py`（我的路径是`.local/lib/python3.8/site-packages/textrank4zh/util.py`）第22句`sentence_delimiters = ['?', '!', ';', '？', '！', '。', '；', '……', '…', '\n']`直接改成`sentence_delimiters = ['\n']`。此外，我准备接下来直接去掉对textrank4zh包的依赖，所以不专门在此之前对这一问题进行其他的处理了。  
暂不支持自动分句功能，要求原文中即含分句token。

首先需要安装textrank4zh包及其所需的前置包。建议使用的脚本代码（Linux端）：
```
pip install jieba
pip install numpy
pip install networkx[default]
git clone https://github.com/letiantian/TextRank4ZH.git
cd TextRank4ZH
python setup.py install --user
```

直接调用textrank.py即可运行。  
入参：  
`--dataset_folder_path`&emsp;&emsp;内含test.src的数据集文件夹（如果`selected_num`小于等于0，则还需要有train.tgt）（支持绝对和相对路径）  
`--result_folder_path`&emsp;&emsp;输出值将打印在该文件夹下的`result_file_name`文本文件中（支持绝对和相对路径）  
`--result_file_name`&emsp;&emsp;输出文件名  
`--selected_num`&emsp;&emsp;选择几句话作为输出。如果该值大于0，将选择该值数目；反之，则以训练集样本摘要平均句子数目为该值数目

工作计划和项目日志：
- [ ] 增加对自动分句功能的支持
- [ ] 增加输出结果示例
- [ ] 增加对`dataset_folder_path`和`result_folder_path`使用不严格文件夹路径写法的容错支持
- [ ] 脱离对textrank4zh包的需求，完全重写
- [ ] 增加对`selected_num`超出原文句数情况的考虑