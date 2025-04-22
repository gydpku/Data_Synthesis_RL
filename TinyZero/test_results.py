import os
from datasets import load_from_disk
import time
from verl.utils.reward_score import mednli,logiqa,gsm8k,dentist_qa
import datasets
import re
import pdb
import torch
import wandb
wandb.login(key='6ed283938a8d9f6896f0145553a1cbdaf482482e')
from vllm import LLM,SamplingParams

def get_names(path):
    return [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path,folder))]

def _select_rm_score_data_fn(task):
    if task == 'gsm8k':
        return gsm8k.compute_score, gsm8k.load_data
    elif task == 'dentist_qa':
        return dentist_qa.compute_score, dentist_qa.load_data
    elif task == 'mednli':
        return mednli.compute_score, mednli.load_data
    elif task == 'logiqa':
        return logiqa.compute_score, logiqa.load_data
    else:
        raise NotImplementedError
def load_model(model_name,temperature=0.0,max_new_tokens=1500,cuda_num=0):
    sampling_params = SamplingParams(temperature=temperature,max_tokens=max_new_tokens, top_p=0.95)
# load model
    llm = LLM(model_name,max_model_len=8192,gpu_memory_utilization=0.95,device="cuda:{0}".format(cuda_num))
    return llm,sampling_params

def generate(prompt,llm,sampling_params):
    responses = llm.generate(
        prompt,
        sampling_params,
    )
    return [response.outputs[0].text for response in responses] #response.choices[0].message.content
instruction = 'Your output thinking process and answer should be enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> thinking process here </think> <answer> the correct answer here </answer>. '
import torch
def clear_cuda_cache():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()
def clean_model_cache():
    import gc
    import time          
    gc.collect()  # Run garbage collection to free up memory
    clear_cuda_cache()
    time.sleep(5)
def calculate_dataset(test_data,llm,sampling_params,score_f):
    #test_dataset=dataset['test']
    num=0
    count=0
    skip_num=50
    #test_data=[data for data in test_dataset]
    for i in range(0,len(test_data),skip_num):
        datas=test_data[i:i+skip_num]
        #'Context: '+' '.join(data['rag'])+
        print('data',datas[0])
        queries=[data['input'] for data in datas]
        responses=generate(queries,llm,sampling_params)
        def get_gold(data):
            try:
                gold=data['std']
            except:
                gold=data['answer']
            return gold
        golds=[get_gold(data) for data in datas]
        def get_pred(response):
            answer_pattern = r'<answer>(.*?)</answer>'
            match = re.finditer(answer_pattern, response)
            matches = list(match)
            if matches:
                final_answer = matches[-1].group(1).strip()
            else:
                final_answer = None
            return final_answer
        preds=[response for response in responses]
        num+=len(preds) #dentist_qa
        count+=sum([score_f(pred,gold,valid=True) for pred,gold in zip(preds,golds)])
    return count,num,count/num
task='dentist_qa'

name_performance={}
'''
model_path='/dccstor/obsidian_llm/yiduo/summary/src/TinyZero/model/Qwen2.5-7B' #os.path.join(path,name)
llm,sparam=load_model(model_path)
score_f,data_f=_select_rm_score_data_fn(task)
_,test_data=data_f() #dataset=datasets.load_from_disk('/dccstor/obsidian_llm/yiduo/summary/src/medical_qa')
count,num,acc=calculate_dataset(test_data,llm,sparam,score_f) 
import pdb 
pdb.set_trace()
'''
path='/dccstor/obsidian_llm/yiduo/summary/src/TinyZero/checkpoints/TinyZero/dentist_qa-Qwen2.5-7B-grpo-temperature-1.2-rollout-16-batchsize-4-response_length-1024/actor'.format(task)
names=get_names(path)
names = sorted(names, key=lambda name: int(re.findall(r'\d+\.?\d*', name)[0]))
wandb.init(project="TinyZero", name="{0}".format(task), entity="guoyd1999-peking-university")
for name in names:
    model_path=os.path.join(path,name)
    try:
        llm,sparam=load_model(model_path)
    except:
        continue
    score_f,data_f=_select_rm_score_data_fn(task)
    _,test_data=data_f() #dataset=datasets.load_from_disk('/dccstor/obsidian_llm/yiduo/summary/src/medical_qa')
    count,num,acc=calculate_dataset(test_data,llm,sparam,score_f)
#    import pdb
 #   pdb.set_trace()
    del llm
    clean_model_cache()
    print('acc',acc)
    pdb.set_trace()
    name_performance[name]=acc
    wandb.log({
        "accuracy": acc
    },step=int(re.findall(r'\d+\.?\d*', name)[0]))
    print(name_performance)
wandb.finish()
pdb.set_trace()
wandb.finish()


