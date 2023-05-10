本文件夹包含一些可以直接调用huggingface.transformers包的Seq2SeqTrainer的模型的通用解决方案。对于不能直接用transformers包的方法，即使使用了transformers包，我也将专门开一个文件夹来储存之（如CPT）  
（在代码中我直接写了只用每个split的前100条数据，因为没算力跑这么多数据。等我有卡可嫖了再说吧）

需要安装transformers包（我用的是4.28.1）、datasets包（我用的是2.12.0）、evaluate包（我用的是0.4.0）、rouge_score包（我用的是0.1.2。我没有测试过这个包的有效性）

我直接调用transformers的Trainer，所以会自动调用所有GPU。可以在Python命令行前添加`CUDA_VISIBLE_DEVICES=3`以设置GPU  
可以在Python命令行前添加`WANDB_MODE=offline`以使wandb不要同步

作者在自己测试代码的过程中，发现可以直接实现运行的模型有：
- <https://huggingface.co/fnlp/bart-base-chinese>