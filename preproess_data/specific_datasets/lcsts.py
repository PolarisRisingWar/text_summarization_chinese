#本代码文件用于直接将下载后的原始LCSTS.zip文件转换为本项目所需的数据格式
#需要提前下载LAC包（我使用的版本是2.1.2）
#对入参介绍见对应的注释

import argparse,os,zipfile
from LAC import LAC

parser = argparse.ArgumentParser()

parser.add_argument("--original_zip_path",default='LCSTS.zip',type=str)
#原始LCSTS.zip文件路径

parser.add_argument("--temp_path",default='./tmp_lcsts',type=str)
#储存LCSTS.zip的临时文件夹。如路径不存在将自动创造，代码运行后将自动删除；如路径不是文件夹将报错
#代码运行后将自动删除文件夹中的文件

parser.add_argument("--tsc_data_path",default='./lcsts_tsc_format',type=str)
#储存处理后的12个文件的路径。如路径不存在将自动创造，代码运行后将自动删除；如路径不是文件夹将报错

args = parser.parse_args()
arg_dict=args.__dict__

lac = LAC(mode='seg')

#解压文件
zFile = zipfile.ZipFile(arg_dict['original_zip_path'], "r")
temp_dir=arg_dict['temp_path']
delete_dir=False  #运行结束后是否需要删除temp文件夹
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
    delete_dir=True
else:
    if os.path.isfile(temp_dir):
        raise Exception('temp_dir参数需要是一个文件夹，不能是一个文件！')
for fileM in zFile.namelist(): 
    zFile.extract(fileM,temp_dir)
zFile.close()

#处理文件（耗时约1h30min）
splits_original=['train','valid','test']
splits_after=['train','test','val']

parts=['src','tgt']
after_folder=arg_dict['tsc_data_path']
sentence_split_crition='。（）；:：，?？!！'

for split_index in range(3):  #遍历所有split
    for part in parts:
        this_file_name=temp_dir+'/'+splits_original[split_index]+'.'+part+'.txt'
        original_data=open(this_file_name).readlines()
        
        #未分词版
        after_file_name=after_folder+'/'+splits_after[split_index]+'.'+part
        after_file=open(after_file_name,'w')

        #分词版
        after_tok_file_name=after_folder+'/'+splits_after[split_index]+'_tok.'+part
        after_tok_file=open(after_tok_file_name,'w')

        for sentence in original_data:
            #分词
            tok_sentence=' '.join(lac.run(sentence))

            #分句
            new_sentence=[]
            for c in tok_sentence[:-1]:
                new_sentence.append(c)
                if c in sentence_split_crition:
                    new_sentence.append('</sentence>')
            tokenized_sentence=''.join(new_sentence)
            after_tok_file.write(tokenized_sentence+'\n')
            after_file.write(tokenized_sentence.replace(' ','')+'\n')
        
        after_file.close()
        after_tok_file.close()

#开删
for fileM in zFile.namelist():
    file_path=temp_dir+'/'+fileM
    os.remove(file_path)
if delete_dir:
    os.removedirs(temp_dir)
