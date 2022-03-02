#计算train.tgt中平均句数
#不支持自动分句
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file_path",default='dataset_path/train.tgt',type=str)

args = parser.parse_args()
arg_dict=args.__dict__

train_results=open(arg_dict['file_path']).readlines()
average_num=0
for line in train_results:
    average_num+=len(line.split('</sentence>'))-1
average_num/=len(train_results)
selected_num=max(round(average_num),1)  #至少要选1句
print(selected_num)