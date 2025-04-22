# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""
from concurrent.futures import ProcessPoolExecutor
import random
import pdb
from statistics import mode
import re
import os
import datasets
import pdb
from verl.utils.hdfs_io import copy, makedirs
from openai_call import query_azure_openai_chatgpt_chat
import argparse
from datasets import Dataset, DatasetDict
from vllm import LLM,SamplingParams
from verl.utils.reward_score.gsm8k import compute_score
import torch
def load_model(model_name,temperature=0.0,max_new_tokens=1500):
    llm = LLM(model_name,max_model_len=8192,gpu_memory_utilization=0.95) #device="cuda:1"
    sampling_params = SamplingParams(temperature=temperature,max_tokens=max_new_tokens, top_p=0.95,n=1)
# load model
    return llm,sampling_params
def compute_single_score(response, ground_truth):
    return compute_score(response, ground_truth, valid=True)
def generate(prompt,llm,sampling_params):
    response = llm.generate(
        prompt,
        sampling_params,
    )
    if sampling_params.n>1:
        return [output.text for output in response[0].outputs]
    try:
        return response.choices[0].message.content
    except:
        return [output.text for output in response[0].outputs]
def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    if solution is None:
#        pdb.set_trace()
#        solution_str=solution_str.split('\n\n')[-2]+' '+'4.8'
        numbers = re.findall(r"-?\b\d+(?:[\.,]\d+)?\b", solution_str) #re.findall(r"-?[0-9\.,]+", solution_str) #solution = re.search("(\\-?[0-9\\.\\,]+)", solution_str)
 #       pdb.set_trace()
        assert numbers is not None
        return str(numbers[-1])
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/gsm8k')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    num_few_shot = 5
    data_source = 'openai/gsm8k'
    data_num=1
    dataset = datasets.load_dataset(data_source, 'main')
    dataset_gen=datasets.load_from_disk(f'/dccstor/obsidian_llm/yiduo/summary/src/GSM8k_{data_num}_1000')
    train_subset = dataset['train'].select(range(data_num))
    train_data=[]
    for data in train_subset:
        train_data.append({'input':data['question'],'output':data['answer'],'mode_rate':1.0,'score':16})
    instruction_following = "Let's think step by step and output the final answer after \"####\"."
    def generate_guide_prompt(demo_data):
        examples_text=''
        for data_id,data in enumerate(demo_data):
            examples_text+='Example id {0}: Input:{1} Output:{2}'.format(data_id,data['input'],data['output'])
        return f"Here is some example {examples_text}, Please follow the examples to solve the current question. Current question: "
    # Concatenate the selected subset with the custom dataset
#    pdb.set_trace()
    train_dataset = datasets.concatenate_datasets([Dataset.from_list(train_data), dataset_gen['train']]) #range(int((len(dataset_gen['train'])-(100-data_num))//2),int((len(dataset_gen['train'])+(100-data_num))//2)))]) 
    difficult_datas=[]    
    train_data_num=100
    hard_data=[]
#    model,sample_param=load_model('/dccstor/obsidian_llm/yiduo/summary/src/TinyZero/model/Qwen2.5-7B')
    difficult_datas=torch.load('difficult_datas.pt')
    hard_data=torch.load('hard_data.pt')

    '''
    pdb.set_trace()
    for data in difficult_datas:
        if data['score']:
            prompt="The current sample is overly simplistic and can be solved effortlessly by the model. Please generate an alternative and task-similar sample that presents a significantly more challenging and intricate problem—one that requires multi-step reasoning, creative problem-solving, and deeper analytical thought. Only output the revised sample in the python dictionary form that supporting the further parse operation. Current sample: {{'input':{0},'output':{1}}}".format(data['input'],data['output'])
            response=query_azure_openai_chatgpt_chat(prompt)
            dic_str=response[response.find('{'):response.rfind('}')+1]
#            pdb.set_trace()
            try:
               dic=eval(dic_str)
             
               hard_data.append(dic)
            except:
               continue
        else:
            hard_data.append({'input':data['input'],'output':data['output']})
    
    torch.save(hard_data,'hard_data.pt')
    '''
    train_data=[]
    for data_id,data in enumerate(hard_data):
        print(data_id,len(hard_data))
        prompt="You should give an output to the query and use '### final answer' to end your output."+str(data['input'])+"Let's think step by step."
        responses=query_azure_openai_chatgpt_chat(prompt,temperature=0.7,n=16)
        responses.append(str(data['output']))
        answers=[extract_solution(response) for response in responses]
        mode_value = mode(answers)
        mode_ids = [index for index, value in enumerate(answers) if value == mode_value]
        selected_response=[(responses[index],len(responses[index])) for index in mode_ids]
        train_data.append({'input':str(data['input']),'output':str(selected_response[0][0])})
#        pdb.set_trace()
    torch.save(train_data,'train_data.pt')
    train_dataset=Dataset.from_list(train_data)
    '''
    for data in dataset_gen['train']:
        #pdb.set_trace()
        if len(train_data)>5:
            guide_prompt=generate_guide_prompt(random.choices(train_data,5))
        else:
            guide_prompt=generate_guide_prompt(train_data)
        responses=generate(data['input']+ ' ' + instruction_following,model,sample_param)
        ground_truth=extract_solution(data['output'])
        
        with ProcessPoolExecutor() as executor:
            gathered_scores = list(executor.map(compute_single_score, responses, [ground_truth]*len(responses)))
        
        data['score']=sum(gathered_scores)
        difficult_datas.append(data)
    difficult_datas.sort(key=lambda x:x['score'])
    torch.save(difficult_datas,'difficult_datas.pt')
    pdb.set_trace()
    '''
    '''
    difficult_datas=difficult_datas[:train_data_num]
    difficult_datas.extend(train_data)
    
    train_dataset=Dataset.from_list(difficult_datas)
    '''
    '''
    test_data=[]
    for data in dataset['test']:
        if len(train_data)>5:
            guide_prompt=generate_guide_prompt(random.choices(train_data,5))
        else:
            guide_prompt=generate_guide_prompt(train_data)
        data['question']=guide_prompt+data['question']
        test_data.append(data)
    difficult_datas.extend(train_data)
    train_dataset=Dataset.from_list(difficult_datas)
#    pdb.set_trace()
    '''
    
#train_dataset = dataset['train'].select(range(5)).concatenate(dataset_gen['train']) #dataset.concatenate([dataset['train'].select(range(5)),dataset_gen])
    test_dataset = dataset['test'] #Dataset.from_list(test_data) #dataset['test']
    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
           # question_raw = example.pop('question')
            try:
                question_raw = example.pop('question')
                question = question_raw + ' ' + instruction_following
            except:
                question_raw = example.pop('input')
                question = question_raw + ' ' + instruction_following
            try:
                answer_raw = example.pop('answer')
            except:
                answer_raw = example.pop('output')
            if not answer_raw:  
                answer_raw = example.pop('output') 
            solution = extract_solution(answer_raw)
            
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer_raw,
                    "question": question_raw,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
#    pdb.set_trace()
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
#    pdb.set_trace()
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
