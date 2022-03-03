# TransformerSum

使用[HHousen/TransformerSum: Models to perform neural summarization (extractive and abstractive) using machine learning transformers and a tool to convert abstractive summarization datasets to the extractive task.](https://github.com/HHousen/TransformerSum)实现。  
暂不支持自动分句功能，要求原文中即含分句token。

首先需要按照TransformerSum原项目提供的方式来克隆并运行项目，推荐的终端运行脚本（Linux平台）为：
```
git clone https://github.com/HHousen/TransformerSum.git
cd TransformerSum
conda env create --file environment.yml
```
这样会自动根据环境的需求新建anaconda虚拟环境transformersum，使用该环境即可直接运行TransformerSum的代码。

原TransformerSum仅提供了英文数据微调后的预训练模型，因此无法直接在中文数据上直接进行推理。本项目后期将使用一些中文数据微调得到预训练模型并进行发布，目前仅支持直接使用TransformerSum进行微调操作。  
具体细节分别参照抽取式任务（[extractive](./extractive/)）和生成式任务（[abstractive](./abstractive/)）目录下的README文件。

工作计划和项目日志：
- [ ] 增加对自动分句功能的支持
- [ ] 增加输出结果示例
- [ ] 脱离对TransformerSum项目的依赖，重写代码
