import argparse
from transformers import pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--data_file_path",default='unified_data/example1/test.src',type=str)
parser.add_argument("--result_file_path",default='mypath/result.txt',type=str)
parser.add_argument("--pretrained_model_or_path",default='csebuetnlp/mT5_multilingual_XLSum',type=str)

args = parser.parse_args()
arg_dict=args.__dict__

summarizer = pipeline("summarization",arg_dict['pretrained_model_or_path'])
srcs=open(arg_dict['data_file_path']).readlines()

result_file=open(arg_dict['result_file_path'],'w')
summary_result=summarizer([x.strip().replace('</sentence>','') for x in srcs])
result_file.writelines(x['summary_text']+'\n' for x in summary_result)
result_file.close()
