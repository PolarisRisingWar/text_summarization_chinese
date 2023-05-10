#本代码文件用于直接将下载后的原始LCSTS.zip文件转换为本项目所需的数据格式
#需要提前下载LAC包（我使用的版本是2.1.2）
#对入参介绍见对应的注释

import argparse,os,zipfile,json,shutil
from LAC import LAC

parser = argparse.ArgumentParser()

parser.add_argument("--original_zip_path",default='LCSTS.zip',type=str)
#原始LCSTS.zip文件路径

parser.add_argument("--temp_dir",default='./tmp_lcsts',type=str)
#储存LCSTS.zip解压缩文件的临时文件夹。如路径不存在将自动创造，代码运行后将自动删除；如路径不是文件夹将报错
#代码运行后将自动删除文件夹中的文件

parser.add_argument("--tsc_data_path",default='./lcsts_tsc_format',type=str)
#储存处理后的3个文件的路径。如路径不存在将自动创造；如路径不是文件夹将报错

parser.add_argument("--tokenize",action="store_true",default=False)  #是否直接进行分词
parser.add_argument("--sent_split",action="store_true",default=False)  #是否直接进行分句

args=parser.parse_args()
arg_dict=args.__dict__

lac=LAC(mode='seg')

#解压文件
zFile=zipfile.ZipFile(arg_dict['original_zip_path'],"r")
temp_dir=arg_dict['temp_dir']

after_folder=arg_dict['tsc_data_path']

delete_dir=False
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
    delete_dir=True  #如果这个文件夹位置本来就是空的，那就连着文件夹一起删了

for folder in [temp_dir,after_folder]:
    if os.path.isfile(folder):
        raise Exception(folder+'需要是一个文件夹，不能是一个文件！')
    
for fileM in zFile.namelist(): 
    zFile.extract(fileM,temp_dir)
zFile.close()

#处理文件
splits=['train','valid','test']

sentence_split_crition='。（）；:：，?？!！'

for split_index in range(3):  #遍历所有split
    after_file=open(after_folder+'/'+splits[split_index]+'.jsonl','w')
    
    src_file_name=temp_dir+'/'+splits[split_index]+'.src.txt'
    original_src=open(src_file_name).readlines()
    tgt_file_name=temp_dir+'/'+splits[split_index]+'.tgt.txt'
    original_tgt=open(tgt_file_name).readlines()

    for sample in zip(original_src,original_tgt):
        #TODO: 分词相关（之前几版写过代码）
        after_file.write(json.dumps({'src':sample[0],'tgt':sample[1]},ensure_ascii=False)+'\n')
        
    after_file.close()

#开删
for fileM in zFile.namelist():
    file_path=temp_dir+'/'+fileM
    if os.path.isfile(file_path):
        os.remove(file_path)
    if os.path.isdir(file_path):
        shutil.rmtree(file_path)
if delete_dir:
    os.removedirs(temp_dir)
