import multiprocessing
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import gc
import torch
import time
from eval.tasks.task_manager import TaskManager
import pdb
from typing import List, Dict, Tuple
from vllm import LLM, SamplingParams
from transformers import PreTrainedModel

def load_model_and_infer(model_path: str,data_batch: List[Dict],task,temperature=0.0,max_tokens=600,top_p=0.95):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    trained_model=LLM(model=model_path,gpu_memory_utilization=0.95)
    sampling_params = SamplingParams(temperature=temperature,max_tokens=max_tokens, top_p=top_p)
    batch_prompts = [data['Input'] for data in data_batch]
    if 'gsm8k' in task.lower():
        batch_prompts = [data['Input']+"\n Let's think step by step. At the end, you MUST write the answer as an number after '####' likes '#### number'." for data in data_batch]
    elif 'sql' in task.lower():
        batch_prompts = [data['Input'].replace('SELECT','')+'To generate the SQL query to' for data in data_batch] 
    else:
        batch_prompts = [data['Input']+"\n End your response with 'The final answer is xxx'." for data in data_batch] 
    outputs = trained_model.generate(batch_prompts, sampling_params)
    return outputs
              
def valid_results_collect(model_path: str, valid_data: List[Dict], task: str) -> Tuple[List[Tuple], List[Tuple]]:
    outputs=load_model_and_infer(model_path,valid_data,task)
    task_evaluator = TaskManager()

    # Load task1 and bind its functions to the manager
    task_evaluator.load_task(task)
    failed_cases=[]
    correct_cases=[]
    for data, output in zip(data_batch, outputs):
            predicted_output = output.outputs[0].text
            pred=task_evaluator.process_prediction(predicted_output, task)
            label=task_evaluator.process_label(data['Output'], task)
            eval_result=task_evaluator.eval_function(pred, label, task)
            if eval_result is True:
                correct_cases.append((data['Input'],predicted_output,label,data))
            else:
                failed_cases.append((data['Input'],predicted_output,label,data))
    del trained_model
    gc.collect()  # Run garbage collection to free up memory
    clear_cuda_cache()
    time.sleep(5)
    return failed_cases,correct_cases
def clear_cuda_cache():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()
def label_transform(label: int) -> str:
    if label==1:
        return 'neutral'
    if label==0:
        return 'entailment'
    if label==2:
        return 'contradiction'
def sql_evaluation(trained_model: PreTrainedModel, valid_data: List[Dict], dp_path_str: str='/dccstor/obsidian_llm/yiduo/AgentBench/DAMO-ConvAI/bird/data/train/train_databases') -> Tuple[List[Tuple],List[Tuple]]:
    id=0
    failed_cases=[]
    correct_cases=[]
    sampling_params = SamplingParams(temperature=0.0,max_tokens=600, top_p=0.95)
    for triple in valid_data:
        db_id,prompt,ground_truth=triple
        prompt=prompt.replace('SELECT','')
        db_path=f'{dp_path_str}/{db_id}/{db_id}.sqlite'
        prompt+=' To generate the SQL query to'
        output=trained_model.generate(prompt, sampling_params) 
        predicted_sql = output[0].outputs[0].text
        
        prior_pred=predicted_sql.split('final SQL')[0]
        try:
            predicted_sql = predicted_sql.split('final SQL')[1].strip()
        except:
            predicted_sql = 'SELECT'+predicted_sql.split('SELECT')[1]
        predicted_sql=predicted_sql.split(';')[0]
        predicted_sql=predicted_sql[predicted_sql.find('SELECT'):] #[1:]
        conn=sqlite3.connect(db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(predicted_sql)
            predicted_res = cursor.fetchall()
            cursor.execute(ground_truth)
            ground_truth_res = cursor.fetchall()
            if set(predicted_res) != set(ground_truth_res):
                failed_cases.append((id,prompt,prior_pred+predicted_sql,valid_data[id],ground_truth,predicted_res,ground_truth_res))
            else:
                correct_cases.append((id,prompt,prior_pred+predicted_sql,valid_data[id],ground_truth,predicted_res,ground_truth_res))
        except Exception as e:
            failed_cases.append((id,prompt,predicted_sql,valid_data[id],ground_truth,str(Exception)+str(e)))
        return failed_cases,correct_cases
    
def safty_float_check(pred: str, label: str) -> bool:
    if pred is None:
        return True
    if label is None:
        return False
    if abs(float(pred)-float(label))>1e-3:
        return True
    return False
