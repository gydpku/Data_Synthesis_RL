import pdb
from statistics import mode
import torch
import numpy as np
import re
import random
import os
import random
import json
import requests
from typing import List, Dict
from eval.tasks.task_manager import TaskManager
from model_inference.batch_inference import batch_inference
from retriever.BM25_retriever import search_relevant_documents
from datasets import Dataset,DatasetDict,load_dataset
from datasets import Dataset, DatasetDict
from model_inference.openai_call import query_azure_openai_chatgpt_chat


seed=2024
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(seed)
META_PROMPT = """
As a DatasetGenerator, your task is to generate one new example (`input` and `output`) based on the [new instruction], [reference passage], and [few-shot examples]. Please provide a JSON dictionary response that includes the new `input` and its corresponding `output`. Use the `input` and `output` keys in the dictionary.
Try you best to ensure that the input and output you generate are distinct from the provided examples while maintaining a diverse, detailed, precise, comprehensive, and high-quality response.
"""
META_PROMPT_simple = """
As a DatasetGenerator, your task is to generate one new examples (`input` and `output`) based on the [new instruction] and [few-shot examples]. Please provide a JSON dictionary response that includes the new `input` and its corresponding `output`. Use the `input` and `output` keys in the dictionary.
Try you best to ensure that the input and output you generate are distinct from the provided examples while maintaining a diverse, detailed, precise, comprehensive, and high-quality response.
"""

def example_check(new_example:str) -> Dict:
        if not new_example:
            return
        new_example=new_example[new_example.find('{'):new_example.rfind('}')+1]
        try:
            return eval(new_example)
        except:
            try:
                import json
                return json.loads(new_example)
            except:
                try:
                    new_example=extract_examples(new_example)[0]
                except:
                    return
        return new_example
def label_transform(label: int ) -> str:
    if label==1:
        return 'neutral'
    if label==0:
        return 'entailment'
    if label==2:
        return 'contradiction'
def aug_few_shot_prompt_pattern_generate(high_quality_examples: List[Dict], task_instruction: str, obj_passage: List[str],output_instruction:str) -> str:   
    template=META_PROMPT
    #knowledge_prompt=f'extract and list the domain knowledge from the passage {obj_passage[:min(2048,len(obj_passage))]} that is related to the task instruction {task_instruction}. You should only output the knowledge without any other text. The knowledge is:'
    template+='You must consider the task instruction (task knowledge), and the passage (domain knowledge) to generate your training data.'
    template+=""" Here is the task instruction:{0}\n""".format(task_instruction)
    template+=""" Here is the output instruction:{0}\n. You should follow the output format in the instruction strictly to generate data!!!""".format(output_instruction)
    #template+=f"""Here is the sample pattern {pattern}. You should follow the input and output pattern strictly to generate data!!!""" 
    #pdb.set_trace()
    '''
    template+=""" You can refer to the provided examples. You should generate examples that are in the same difficulty or are harder. """ 

    for id in range(len(high_quality_examples)):
        template+='Demo Example {0}: {1}'.format(id,high_quality_examples[id])
    '''
    template+=" Here is some related knowledge passage that you must refer to. Your generated example must base on the knowledge/information of the passage." 
    template+="Related Objects or Passages:{0}".format(obj_passage[:min(2048,len(obj_passage))])
    new_examples=[]
    template+="Before generating the new example, ensure that you strictly adhere to the rules mentioned in the [Requirement] and follow the format of the [high-quality examples]. Think twice before generating a new example. New example (in JSON):"
    
    return query_azure_openai_chatgpt_chat(template,temperature=0.7)
def cot_check_fill(demo_examples: List[Dict], task_name: str = '') -> List[Dict]:
    first_example=demo_examples[0]
    demo_cot_examples=[]
    prompt='Does the example has the specific solution (step by step) to get its final output? Directly output yes or no.'
    prompt+='Example:{0}'.format(first_example)
    cot_status=query_azure_openai_chatgpt_chat(prompt)
    if 'yes' in cot_status.lower() or 'gsm8k' in task_name.lower():
        return demo_examples
    else:
        for example in demo_examples:
            prompt="Your task is to generate the specific solution (step by step) to get its final output. The solution starts with 'Let's think step by step' and ends with 'The final answer is ...'. Example:{0}. Directly and only output the text solution without the input.".format('Input: '+example['Input']+'Output: '+example['Output'])
            cot_solution=query_azure_openai_chatgpt_chat(prompt)
 #           pdb.set_trace()
            demo_cot_examples.append({'Input':example['Input'],'Output':cot_solution})

        return demo_cot_examples
def data_sample_pattern(instruction: str, domain: str, num: int, store_name: str, demo_examples: List[Dict], paths: List[str], harder: bool=False, simpler: bool=False,temperature: float=0.7, task_name: str='nli', voting: bool=False, pattern: bool=False, iteration_number: int=1, sample_demo_num: int=3, passage_num: int=5000, valid_num: int=100) -> List[Dict]:
    task_instruction=instruction
    print('load_cot_data...')
    demo_cot_examples=demo_examples
    
    print(demo_cot_examples[0],'\n')
    voting=True #    print(demo_examples[0],'\n')

    print('store_name','{0}.pt'.format(store_name))
    import os
    cur_path=os.getcwd()
    if not harder and not simpler:
        print('Retrieving passage ...')
        
        try:
            passages=torch.load(os.path.join(cur_path,'src','{0}.pt'.format(domain))) #'medical_passages_sub.pt')
        except:
            passages=search_relevant_documents(domain,10000,paths) 
            torch.save(passages,os.path.join(cur_path,'src','{0}.pt'.format(domain))) #'medical_passages_sub.pt')
        import random
        
        passages=list(set(passages))
        random.shuffle(passages)
        print('passages',len(passages))
        try:
            synthetic_examples=torch.load(os.path.join(cur_path,'src','{0}.pt'.format(store_name)))
        except:
            synthetic_examples=[]
        start=len(synthetic_examples)
        print('Generating data, which starts with',start,'ends with',num+3)
        if start>num:
            return synthetic_examples[:num]
        #pdb.set_trace()
        sampled_demo_examples=random.sample(demo_cot_examples, min(5,len(demo_cot_examples)))
        pattern_prompt=f"""The task is {instruction}. Your task is to summarize the task's input and output parse format in general. You can refer to some demonstration examples of this task: {sampled_demo_examples}. You should output the input format first and then the output format."""
        pattern=query_azure_openai_chatgpt_chat(pattern_prompt)
        
        examples_str=[]
        task_evaluator = TaskManager()
        task_evaluator.load_task(task_name)
        output_instruction=task_evaluator.get_output_instruction()
        for num_id in range(start,num+3):
            
            sample_size = 1 # random.randint(1,5)
            sampled_objects = passages[num_id]
            sampled_demo_examples=random.sample(demo_cot_examples, min(sample_demo_num,len(demo_cot_examples)))
            examples_str.append(aug_few_shot_prompt_pattern_generate(sampled_demo_examples,task_instruction,sampled_objects,output_instruction))
        #pdb.set_trace()
        format_checked_examples=[]
        for example in examples_str:
            format_prompt = f"""Considering the task pattern: {pattern}, your task is to revise the input and output format of {example} to match the format of the high-quality examples: {sampled_demo_examples}. Output only the revised example, without any additional text."""
            revised_example=query_azure_openai_chatgpt_chat(format_prompt)
            format_checked_examples.append(revised_example)
            #pdb.set_trace()
        results=format_checked_examples
        synthetic_examples=precise_check(results,synthetic_examples,task_name,voting=voting)
        torch.save(synthetic_examples,os.path.join(cur_path,'src','{0}.pt'.format(store_name)))
    elif harder:
        try:
            return torch.load(os.path.join(cur_path,'src','{0}_harder.pt'.format(store_name)))
        except:
            synthetic_examples=[]
        prompts=[]

        for data in demo_examples:
            prompt="""The current sample is overly simplistic and can be solved effortlessly by the model. Please generate an alternative and task-similar sample that presents a significantly more challenging 
            and intricate problem—one that requires multi-step reasoning, creative problem-solving, and deeper analytical thought. Only output the revised sample in the python dictionary form. Current sample:{0}""".format(data)
            prompts.append(prompt)
        results=[]
        for prompt in prompts:results.append(query_azure_openai_chatgpt_chat(prompt,temperature=0.7))
        synthetic_examples=precise_check(results,synthetic_examples,task_name,voting=voting)
        torch.save(synthetic_examples,os.path.join(cur_path,'src','{0}_harder.pt'.format(store_name)))
    else:
        try:
            return torch.load(os.path.join(cur_path,'src','{0}_simpler.pt'.format(store_name)))
        except:
            synthetic_examples=[]
        prompts=[]
        for data in demo_examples:
            prompt="""The current sample is too hard and can not be solved by the model. Please generate an alternative and task-similar sample that presents a simpler sample or a sub-problem of the original sample. Only output the revised sample in the python dictionary form. Current sample:{0}""".format(data)
            prompts.append(prompt)
        results=[]
        for prompt in prompts:results.append(query_azure_openai_chatgpt_chat(prompt,temperature=0.7))
        synthetic_examples=precise_check(results,synthetic_examples,task_name,voting=voting)
        torch.save(synthetic_examples,os.path.join(cur_path,'src','{0}_simpler.pt'.format(store_name)))
 
    return synthetic_examples[:num] #[example_check(result) for result in results][:num] #examples=[example_check(result) for result in results][:num]

def precise_check(results,synthetic_examples: List[Dict],task_name: str, voting: bool=False):
    new_examples=[example for example in (example_check(result) for result in results) if example]
    if voting:
        vot_examples=[]
        for example_id,example in enumerate(new_examples):
            print(example_id)
            short_output,random_output,long_output=majority_voting(example,task_name)
            if random_output:
                vot_examples.append({'input':example['input'],'output':random_output})
        new_examples=vot_examples
    synthetic_examples.extend(new_examples)
    return synthetic_examples

def majority_voting(example: Dict, task_name: str, n: int=4) -> tuple[Dict, Dict, Dict]:
    example_input=example['input']
    output=example['output']
    task_evaluator = TaskManager()
    task_evaluator.load_task(task_name)
    output_instruction=task_evaluator.get_output_instruction()
    prompt=str(example_input)+output_instruction
    '''
    if 'nli' in task_name.lower():
        prompt="You should give an output to the query and use 'The final answer is xxx' to end your output."+input+"Let's think step by step."
    elif 'gsm8k' in task_name.lower():
        prompt="You should give an output to the query and use '### final answer' to end your output."+input+"Let's think step by step."
    else:
        prompt="You should give an output to the query and use 'The final answer is xxx' to end your output."+input+"Let's think step by step."
    '''
    responses=query_azure_openai_chatgpt_chat(prompt,temperature=0.7,n=n)
    responses.append(output)
    

    # Load task1 and bind its functions to the manager
    answers = []
    for response in responses:
        result = task_evaluator.process_prediction(response)
        if result is not None:
            answers.append(result)
    #import pdb
    #pdb.set_trace()
    #answers=[task_evaluator.process_prediction(response) if task_evaluator.process_prediction(response) for response in responses] 
    try:
        mode_value = mode(answers)
    except:
        return None, None, None, None
    mode_ids = [index for index, value in enumerate(answers) if value == mode_value]
    selected_response=[(responses[index],len(responses[index])) for index in mode_ids]
    selected_response.sort(key=lambda x:x[1])
    return selected_response[0][0],responses[random.choice(mode_ids)],selected_response[-1][0]
def clean_input(input: str) -> str:
    if 'output' in input:
        input=input.split('output')[0]
    elif 'Output' in input:
        input=input.split('Output')[0]
    return input.strip()
def clean_output(output: str) -> str:
    if 'output' in output:
       output=output.replace('output','').replace(':','') #.split('output')[1]
    elif 'Output' in output:
        output=output.replace('Output','').replace(':','') #output.split('Output')[1]
    return output.strip()
def clean_and_collect_dataset(examples: List[Dict], instruction: str, name: str):
    if isinstance(examples[0],str):
        datas=[eval(ele[ele.find('{'):ele.find('}')+1]) for ele in examples]
    else:
        datas=examples
    train_data=[{'instruction':instruction+' '+clean_input(item[list(item.keys())[0]]),'output':clean_output(item[list(item.keys())[1]])} for item in datas]
    dataset=Dataset.from_dict({key: [str(dic[key]) for dic in train_data] for key in train_data[0]})
    dataset=dataset.shuffle(seed=2022)
    dataset.save_to_disk(name)
def extract_examples(text: str) -> List[Dict]:
    splits=text.split('}')
    raw_examples=[]
    for split in splits: 
        if 'input' in split or 'Input' in split:
            raw_examples.append(split)
    examples=[]
    for raw_example in raw_examples:
        input_tag='"Input":' if '"Input":' in str(raw_example) else "'Input':" #try:
        #    print(raw_example) #pdb.set_trace()
        input_output=str(raw_example).split(input_tag)[1]
 #       except:
#            pdb.set_trace()
        if 'output' in input_output:
            output_tag='"output":' if '"output":' in str(input_output) else "'output':"
            input,output=input_output.split(output_tag)[0],input_output.split(output_tag)[1]
        elif 'Output' in input_output:
            output_tag='"Output":' if '"Output":' in str(input_output) else "'Output':"
            input,output=input_output.split(output_tag)[0],input_output.split(output_tag)[1]
        else:
            continue
            pdb.set_trace()
        new_example={'input':input.replace('{','').replace('"','').replace('\n','').strip(),'output':output.replace(':','').replace('"','').replace('}','').replace('\\n','\n').replace('\\', ' ').strip()}
        examples.append(new_example)
    return examples
