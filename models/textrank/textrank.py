import argparse
from textrank4zh import TextRank4Sentence

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_folder_path",default='preproess_data/example_dataset',type=str)
parser.add_argument("--result_folder_path",default='./output',type=str)
parser.add_argument("--result_file_name",default='result.txt',type=str)
parser.add_argument("--selected_num",default=3,type=int)

args = parser.parse_args()
arg_dict=args.__dict__

#计算需要选择几句话
if arg_dict['selected_num']>0:
    selected_num=arg_dict['selected_num']
else:
    train_results=open(arg_dict['dataset_folder_path']+'/train.tgt').readlines()
    average_num=0
    for line in train_results:
        average_num+=len(line.split('</sentence>'))-1
    average_num/=len(train_results)
    selected_num=max(round(average_num),1)  #至少要选1句

#通过textrank4zh包计算得到结果
original_texts=open(arg_dict['dataset_folder_path']+'/test.src').readlines()
result_file=open(arg_dict['result_folder_path']+'/'+arg_dict['result_file_name'],'w',
                encoding='utf-8')
for d in original_texts:
    tr4s = TextRank4Sentence()
    sentence_list=d.split('</sentence>')[:-1]
    text='\n'.join(sentence_list)
    tr4s.analyze(text=text, lower=True, source = 'all_filters')
    pl=[]
    for item in tr4s.get_key_sentences(num=selected_num):
        pl.append(item.index)
    pl.sort()
    result=''.join([sentence_list[x] for x in pl])
    result_file.write(result+'\n')
result_file.close()
