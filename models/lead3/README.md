# LEAD-3

返回测试集样本原文的前N句话（原版N=3）。暂不支持自动分句功能，要求原文中即含分句token。

直接调用lead3.py即可运行。

入参：  
`--dataset_folder_path`&emsp;&emsp;内含test.src的数据集文件夹（如果selected_num小于等于0，则还需要有train.tgt）（支持绝对和相对路径）  
`--result_folder_path`&emsp;&emsp;输出值将打印在该文件夹下的result_file_name文本文件中（支持绝对和相对路径）  
`--result_file_name`&emsp;&emsp;输出文件名  
`--selected_num`&emsp;&emsp;选择几句话作为输出。如果该值大于0，将选择该值数目；反之，则以训练集样本摘要平均句子数目为该值数目  

工作计划和项目日志：
- [ ] 增加对自动分句功能的支持
- [ ] 增加输出结果示例
- [ ] 增加对dataset_folder_path和result_folder_path使用不严格文件夹路径写法的容错支持
- [ ] 增加对selected_num超出原文句数情况的考虑