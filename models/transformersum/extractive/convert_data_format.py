#将本项目所采用的原始数据格式转换为TransformerSum抽取式任务所需的格式

import argparse,json

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_folder_path",default='preproess_data/example_dataset',type=str)
parser.add_argument("--output_folder_path",default='./output',type=str)
parser.add_argument("--dataset_name",default='example1',type=str)

args = parser.parse_args()
arg_dict=args.__dict__

for split in ['train','val','test']:
    split_src=open(arg_dict['dataset_folder_path']+'/'+split+'.src').readlines()
    split_tgt=open(arg_dict['dataset_folder_path']+'/'+split+'.tgt').readlines()
    split_label=json.load(open(arg_dict['dataset_folder_path']+'/'+split+'.label'))

    output_file=open(arg_dict['output_folder_path']+'/'+arg_dict['dataset_name']+'.'+split+\
                    '.0.json','w')
    output_object=[]

    for sample_index in range(len(split_src)):  #遍历所有样本
        output_json={}
        sentence_list=split_src[sample_index].strip().split('</sentence>')
        if len(sentence_list[-1])==0:
            sentence_list=sentence_list[:-1]
        output_json['src']=[list(x) for x in sentence_list]
        if split=='test':
            output_json['tgt']=split_tgt[sample_index].strip().replace('</sentence>','<q>')[:-3]
        sentence_length=len(sentence_list)
        sentence_label=[0 for _ in range(sentence_length)]
        for i in split_label[sample_index]:
            sentence_label[i]=1
        output_json['labels']=sentence_label
        output_object.append(output_json)
    json.dump(output_object,output_file,ensure_ascii=False)