import torch.distributed as dist
import numpy as np
import pdb
from typing import List, Dict, Union, Literal
from data_generator.generator import data_sample_pattern
from model_inference.openai_call import query_azure_openai_chatgpt_chat
from vllm import LLM,SamplingParams
from eval.tasks.task_manager import TaskManager
import torch
import subprocess
import random
import argparse
import os
import signal
import subprocess

def stop_vllm_process():
    """Stops the vllm process gracefully."""
    os.kill(os.getpid(), signal.SIGINT)
def reset_cuda_device(gpu_id):
    try:
        # 执行 nvidia-smi --gpu-reset 命令来重置指定的GPU
        subprocess.run(["nvidia-smi", "--gpu-reset", "-i", str(gpu_id)], check=True)
        print(f"CUDA设备 {gpu_id} 已成功重置。")
    except subprocess.CalledProcessError as e:
        print(f"重置CUDA设备 {gpu_id} 时发生错误：{e}")
class DataGenerator:
    """
    Simulates GPT-4o for data generation.
    For real implementation, replace with GPT-4o API calls.
    """
    def __init__(self,task_instruction,demo_examples,domain,multi_task):
        if not multi_task:
            if not domain:
                prompt = f'You can summarize the domain of this task: {task_instruction} into a keyword. You can refer to these task examples {demo_examples}. Only output the keyword'
                domain=query_azure_openai_chatgpt_chat(prompt)
            self.domain = domain
        else:
            if not domain:
                domains=[]
                for instruct,example in zip(task_instruction,demo_examples):
                    prompt = f'You can summarize the domain of this task: {instruct} into a keyword. You can refer to these task examples {example}. Only output the keyword'
                    domain=query_azure_openai_chatgpt_chat(prompt)
                    domains.append(domain)
                domain=domains
            self.domain=domain

    def generate_initial_data(self, demo_examples: List[Dict[str, str]],task_name:str,task_instruction:str, iter,multi_task,passage_paths: List[str],num_examples: int = 500) -> List[Dict[str, str]]:
        """
        Generates initial GSM8K-style problems.
        """
        
        domain=self.domain
        if not multi_task:
            return data_sample_pattern(task_instruction, domain,num_examples, '{0}_{1}_{2}'.format(task_name,num_examples,iter),demo_examples, passage_paths,task_name=task_name)
        else:
            initial_data=[]
            task_num=len(task_instruction)
            if task_name=='math':task_name=[task_name]*task_num
            for sub_domain,sub_task_instruction,sub_examples,sub_task_name in zip(domain,task_instruction,demo_examples,task_name):
                
                sub_data=data_sample_pattern(sub_task_instruction, sub_domain,num_examples//(task_num), '{0}_{1}_{2}_{3}'.format(sub_task_name,num_examples//task_num,iter,sub_domain),sub_examples,passage_paths, task_name=sub_task_name)        
                initial_data.extend(sub_data)
                #pdb.set_trace()
            return initial_data 
        # Inputs:
        #   demo_examples: List[Dict[str, str]] - Demo GSM8K problems (e.g., [{"question": "...", "answer": "...", "calculation": "..."}, ...]).
        #   num_examples: int - Number of initial examples to generate.
        # Outputs:
        #   List[Dict[str, str]] - Generated initial GSM8K problems (each dict: {"question": str, "answer": str, "calculation": str}).
        # Key Parameters:
        #   num_examples: Controls the size of the initial dataset.

    def generate_easier_data(self, hard_data,task_instruction,task_name,iter,multi_task,passage_paths: List[str],) -> List[Dict[str, str]]:
        """
        Generates easier problems based on hard data.
        """
        #collected_data.extend(hard_data)
        if multi_task:
            easier_data=data_sample_pattern(task_instruction, self.domain,len(hard_data), '{0}_simpler_iter_{1}_multi'.format(task_name,iter),hard_data,passage_paths, task_name=task_name,simpler=True)
        else:
            easier_data=data_sample_pattern(task_instruction, self.domain,len(hard_data), '{0}_simpler_iter_{1}'.format(task_name,iter),hard_data,passage_paths, task_name=task_name,simpler=True)
        
        return easier_data
    def generate_hard_data(self, easy_data,task_instruction,task_name,iter,multi_task,passage_paths: List[str],) -> List[Dict[str, str]]:
        """
        Generates harder problems based on easy data.
        """
        #collected_data.extend(hard_data)
        if multi_task:
            harder_data=data_sample_pattern(task_instruction, self.domain,len(easy_data), '{0}_harder_iter_{1}_multi'.format(task_name,iter),easy_data,passage_paths, task_name=task_name,harder=True)
        else:
            harder_data=data_sample_pattern(task_instruction, self.domain,len(easy_data), '{0}_harder_iter_{1}'.format(task_name,iter),easy_data,passage_paths, task_name=task_name,harder=True)
        
        return harder_data
    def generate_diverse_data(self, test_result,task_instruction,task_name,iter,multi_task,passage_paths: List[str],) -> List[Dict[str, str]]:
        """
        Generates harder GSM8K problems based on test results.
        """
        collected_data=[]
        hard_data=[]
        easy_data=[]
        for data in test_result:
            if data[0]:
                easy_data.append(data[1])
            else:
                hard_data.append(data[1])
        #collected_data.extend(hard_data)
        if multi_task:
            harder_data=data_sample_pattern(task_instruction, self.domain,len(easy_data), '{0}_harder_iter_{1}_same_all_test_2_multi'.format(task_name,iter),easy_data,passage_paths, task_name=task_name,harder=True)
        else:
            harder_data=data_sample_pattern(task_instruction, self.domain,len(easy_data), '{0}_harder_iter_{1}_same_all_test_2'.format(task_name,iter),easy_data,passage_paths, task_name=task_name,harder=True)
        if multi_task:
            simpler_data=data_sample_pattern(task_instruction, self.domain,len(hard_data), '{0}_simpler_iter_{1}_same_all_test_2_multi'.format(task_name,iter),hard_data,passage_paths, task_name=task_name,simpler=True)
        else:
            simpler_data=data_sample_pattern(task_instruction, self.domain,len(hard_data), '{0}_simpler_iter_{1}_same_all_test_2'.format(task_name,iter),hard_data,passage_paths, task_name=task_name,simpler=True)
        collected_data.extend(harder_data)
        collected_data.extend(simpler_data)
        #generate_diverse_data collected_data.extend(easy_data)
        return collected_data #hard_data
        # Inputs:
        #   test_result: List[Dict[str, Union[bool, str, Dict[str, str]]]] - Test results from working model (each dict: {"correct": bool, "predicted_answer": str, "reason": str, "example": Dict[str, str]}).
        #   initial_data: List[Dict[str, str]] - Initial data used for testing.
        # Outputs:
        #   List[Dict[str, str]] - Generated hard GSM8K problems (each dict: {"question": str, "answer": str, "calculation": str}).
        # Key Parameters:
        #   test_result: Drives the generation of harder examples based on model's failures.

def get_names(path):
    return [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path,folder))]
class TaskBuffer:
    def __init__(self):
        self.learnable_buffer = []
        self.hard_buffer=[]
        self.easy_buffer=[]
        self.trained_buffer = []
        self.index = -1
    def check_label(self,Tasker):
        new_buffer=[]
        for data in self.learnable_buffer:
          if Tasker.process_label(data['output']):
              new_buffer.append(data)
        self.learnable_buffer=new_buffer

    def calculate_score(self, working_model, tasker,datapoints,n=32,temperature=1.2,self_update=False):
        learnable_buffer = []
        hard_buffer=[]
        easy_buffer=[]
        if self_update:
            self.gather_buffer()
            datapoints=self.learnable_buffer
        for id in range(len(datapoints) // 100 + 1):
            batch=datapoints[id*100:(id+1)*100]
            batch_inputs = [point['input'] for point in batch]
            instruction_following=tasker.get_output_instruction()
            batch_prompt=[str(point['input'])+instruction_following for point in batch]
            batch_labels=[point['output'] for point in batch]
            batch_response=working_model.generate(batch_prompt,n=n,temperature=temperature)
            
            for inp, out, truth in zip(batch_inputs, batch_response, batch_labels):
                #preds = [tasker.process_prediction(pred) for pred in out]
                preds = [pred for pred in out]
                label = tasker.process_label(truth)
                results=[tasker.eval_function(pred, label) for pred in preds]
                score=sum(results)/len(results)
                #pdb.set_trace()
                if score==0:
                    #pdb.set_trace()
                    hard_buffer.append({'input': inp, 'output': truth, 'score': score})
                elif score==1:
                    easy_buffer.append({'input': inp, 'output': truth, 'score': score})
                else:
                    learnable_buffer.append({'input': inp, 'output': truth, 'score': score})
                #new_buffer.append({'input': inp, 'output': truth, 'score': score})
        if self_update:
            self.learnable_buffer=learnable_buffer
            self.hard_buffer=hard_buffer
            self.easy_buffer=easy_buffer
        else:   
            return easy_buffer,learnable_buffer,hard_buffer #self.buffer = new_buffer

    def generate_train_batch(self,train_num,prob_sampling=False,easy_to_hard=False):
        train_dataset = []
        if prob_sampling:
            probabilities=[1-x['score']+1e-3 for x in self.learnable_buffer]
            probabilities = np.array(probabilities)
            probabilities = probabilities / probabilities.sum()
            indices = np.random.choice(len(self.learnable_buffer), size=train_num, replace=True, p=probabilities)
            sampled_data = [self.learnable_buffer[i] for i in indices]
            for i in range(train_num):
                train_dataset.append({'input': sampled_data[i]['input'], 'output': sampled_data[i]['output']})
            return train_dataset
        if easy_to_hard:
            self.learnable_buffer.sort(key=lambda x:-1*x['score'])
        else:
            self.learnable_buffer.sort(key=lambda x:abs(x['score']))
        for i in range(min(train_num,len(self.learnable_buffer))):
            train_dataset.append({'input': self.learnable_buffer[i]['input'], 'output': self.learnable_buffer[i]['output']})
        scores=[data['score'] for data in self.learnable_buffer[:len(train_dataset)]]
        import numpy as np
        try:
            scores_record=torch.load('scores_record.pt')
        except:
            scores_record=[]
        scores_record.append([np.mean(scores),np.std(scores),np.max(scores),np.min(scores),scores])
        torch.save(scores_record,'scores_record.pt')

         #self.index = train_num
        #self.trained_buffer.extend(self.learnable_buffer[:train_num])
        #self.learnable_buffer = self.learnable_buffer[train_num:]
        #train_dataset #.reverse()
        return train_dataset

    def update_buffer(self, data):
        self.learnable_buffer.extend(data)
    def gather_buffer(self):
        self.learnable_buffer.extend(self.hard_buffer)
        self.learnable_buffer.extend(self.easy_buffer)
        self.hard_buffer=[]
        self.easy_buffer=[]


class Trainer:
    """
    A trainer class to execute the shell command
    'sh launch_distributed_project.sh train_7b_base.sh gsm8k_500 1.2 16 4 2048'
    and test if the command has been run.

    Hyperparameter Explanation (defined in __init__):

    - dataset_path (str, default="gsm8k_500"):
        Specifies the path to the dataset used for training. 'gsm8k_500' is indicated as the dataset path.
        This could be a directory or a specific file path depending on how the training script expects the data.

    - temperature (float, default=1.2):
        The temperature parameter used during generation or sampling in the model.
        Temperature influences the randomness of the output.
            - Higher temperature (e.g., > 1): More random and diverse outputs.
            - Lower temperature (e.g., < 1): Less random, more focused and deterministic outputs.
        '1.2' is set as the temperature value.

    - rollout_n (int, default=16):
        'rollout.n' likely refers to the number of rollouts used in a specific training technique,
        possibly related to reinforcement learning or sequence generation.
        '16' is set as the number of rollouts. The exact meaning depends on the
        'train_7b_base.sh' script and the training method it implements.

    - batch_size (int, default=4):
        The batch size used for training. This determines how many examples are processed
        together in each training step.
        '4' is set as the batch size.

    - response_length (int, default=2048):
        Specifies the desired length of the response or output sequences generated by the model.
        '2048' is set as the maximum response length. This limits the length of text generated by the model.

    These parameters are configurations for the training process as defined by the
    'train_7b_base.sh' script. Their specific effects depend on the training script's logic
    and the model architecture it uses.
    """

    def __init__(self,
                 dataset_path="gsm8k_500",
                 model_path="./model",
                 work_dir="./",
                 experiment_name="run_1",
                 temperature=1.2,
                 rollout_n=16,
                 batch_size=4,
                 response_length=2048):
        """
        Initializes the Trainer class with training parameters based on the corrected meanings.

        Args:
            dataset_path (str): Path to the dataset (e.g., "gsm8k_500").
            temperature (float): Temperature for generation/sampling (e.g., 1.2).
            rollout_n (int): Number of rollouts (e.g., 16).
            batch_size (int): Batch size for training (e.g., 4).
            response_length (int): Maximum response length (e.g., 2048).
        """
        self.dataset_path = dataset_path
        self.model_path=model_path
        self.work_dir=work_dir
        self.experiment_name=experiment_name
        self.temperature = temperature
        self.rollout_n = rollout_n
        self.batch_size = batch_size
        self.response_length = response_length
        
        self.command = [
            "sh",
            "./TinyZero/train_base.sh",
            self.dataset_path,
            self.model_path,
            self.experiment_name,
            str(self.temperature),
            str(self.rollout_n),
            str(self.batch_size),
            str(self.response_length)
        ]
        self.run_completed = False # Flag to track if run() was executed

    def run(self):
        """
        Executes the shell command using subprocess.
        """
        import os
        cur_path=os.getcwd()
        
        self.command = [
            "sh",
            os.path.join(cur_path, "TinyZero", "train_base.sh"),
            self.dataset_path,
            self.model_path,
            self.experiment_name,
            str(self.temperature),
            str(self.rollout_n),
            str(self.batch_size),
            str(self.response_length)
        ]
        try:
            print("Executing shell command...")
            process = subprocess.run(
                self.command,
                check=True,  # Raise exception on non-zero exit code
                capture_output=True, # Capture stdout and stderr
                text=True # Decode stdout and stderr as text
            )
            print("Shell command executed successfully.")
            print("--- Command Output (stdout) ---")
            print(process.stdout)
            if process.stderr: # Only print stderr if it's not empty
                print("--- Command Output (stderr) ---")
                print(process.stderr)
            self.run_completed = True # Set flag to True after successful run

        except subprocess.CalledProcessError as e:
            print(f"Error executing shell command:")
            print(f"Return Code: {e.returncode}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            self.run_completed = False # Command failed

        except FileNotFoundError:
            print(f"Error: Could not find 'launch_distributed_project.sh' or 'sh' command. "
                  f"Make sure they are in your system's PATH or the script is in the correct directory.")
            self.run_completed = False

    def test_run(self):
            directory_path=os.path.join(self.work_dir,'checkpoints/TinyZero')
            experiment_path=os.path.join(directory_path,self.experiment_name)
            actor_path=os.path.join(experiment_path,'actor')
            try:
                model_name = get_names(actor_path)
            except:
                print("\nTest failed: run() function was not executed successfully or encountered errors.")
                return False
            if len(model_name)>=1:
                return os.path.join(actor_path,model_name[0])
            else:
                print("\nTest failed: run() function was not executed successfully or encountered errors.")
                return False

            return False
class WorkingModel:
    """
    Simulates Qwen7B for GSM8K task.
    Replace with actual Qwen7B model for real implementation.
    """
    def __init__(self,path,task_name):
        self.trained_data_count = 0
        def clear_cuda_cache():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()

        def clean_model_cache():
            import gc
            import time
            gc.collect()  # Run garbage collection to free up memory
            clear_cuda_cache()
            time.sleep(2)
        clean_model_cache()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        self.llm = LLM(path, max_model_len=8192, gpu_memory_utilization=0.95,device='cuda:0')  # device="cuda:1"
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=1500, top_p=0.95, n=1)

        self.task_name=task_name



    def generate(self,prompt,n,temperature=0.0):
        self.sampling_params = SamplingParams(temperature=temperature, max_tokens=1500, top_p=0.95, n=n)
        response = self.llm.generate(
            prompt,
            self.sampling_params,
        )
        if self.sampling_params.n > 1 and len(prompt)==1:
            return [output.text for output in response[0].outputs]
        if len(prompt)>1 and self.sampling_params.n==1:
            return [res.outputs[0].text for res in response]
        if len(prompt)>1 and self.sampling_params.n>1:
            return  [[output.text for output in res.outputs] for res in response] #[res.outputs[0].text for res in response]
        try:
            return response.choices[0].message.content
        except:
            import pdb
            pdb.set_trace()
            return [output.text for output in response[0].outputs]


    def inference(self, data: List[Dict[str, str]],tasker):
        """
        Performs inference on GSM8K problems.
        """
        # Inputs:
        #   data: List[Dict[str, str]] - List of GSM8K problems to perform inference on (each dict: {"question": str, "answer": str, "calculation": str}).
        # Outputs:
        #   List[Dict[str, Union[bool, str, Dict[str, str]]]] - Test results (each dict: {"correct": bool, "predicted_answer": str, "reason": str, "example": Dict[str, str]}).
        # Key Parameters:
        #   data: The GSM8K problems to be evaluated.
        results=[]
        for id in range(len(data)//100+1):
            batch=data[id*100:(id+1)*100]
            batch_inputs = [point['input'] for point in batch]
            instruction_following=tasker.get_output_instruction()
            batch_prompt=[str(point['input'])+instruction_following for point in batch]
            batch_labels=[point['output'] for point in batch]
            batch_response=self.generate(batch_prompt,n=1)
            
            for input, pred, truth in zip(batch_inputs, batch_response, batch_labels):
                #pred = tasker.process_prediction(pred)
                label = tasker.process_label(truth)
                
                eval_result = tasker.eval_function(pred, label)
                # Append the evaluation result with input/output info to the results list
                results.append((eval_result, {'input': input, 'output': truth}))
            print(id,len(results))
        return results

    def quit(self) -> None:
        """
        Trains the working model on GSM8K hard data.
        """
        def clear_cuda_cache():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()

        def clean_model_cache():
            import gc
            import time
            gc.collect()  # Run garbage collection to free up memory
            clear_cuda_cache()
            time.sleep(20)

        del self.llm
        clean_model_cache()



def update_demo_examples(inference_result,demo_examples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Updates demo examples, similar logic to generic update_demo_examples.
    """
    for data in inference_result:
        if not data[0]:
            demo_examples.append(data[1])
    return demo_examples

    # Inputs:
    #   hard_data: List[Dict[str, str]] - List of hard GSM8K problems.
    #   demo_examples: List[Dict[str, str]] - List of current demo examples.
    #   update_strategy: Literal['True_same', 'False', 'True_increase'] - Strategy for updating demo examples.
    # Outputs:
    #   List[Dict[str, str]] - Updated demo examples.
    # Key Parameters:
    #   update_strategy:  Determines how demo examples are updated (or not updated) based on hard data.



def iterative_training_framework(base_model_path,task_name,task_instruction,demo_examples,dataset_path,work_model_paths,passage_paths,iterations: int = 5,domain=None,multi_task=False) -> WorkingModel:
    """
    Iterative training framework for a task.
    """
    
    data_generator = DataGenerator(task_instruction,demo_examples,domain,multi_task)
    
    Tasker = TaskManager()
    RL_Trainer=Trainer(model_path=base_model_path)
    Tasker.load_task(task_name)
    all_hard_data=[]
    Buffer=TaskBuffer()
    working_model_path=base_model_path
    for iter_num in range(iterations):
        try:
            import os
            cur_path=os.getcwd()
            train_data=torch.load(os.path.join(cur_path,'src','train_data_{0}_{1}.pt'.format(task_name,iter_num)))
        except:
         # 1. Initial Data Generation (GSM8K)       
            initial_data = data_generator.generate_initial_data(demo_examples,task_name,task_instruction,iter_num,multi_task,passage_paths)
            #pdb.set_trace()
            working_model = WorkingModel(base_model_path,task_name)
            Buffer.update_buffer(initial_data) # 2. Inference and Performance Evaluation (GSM8K)
            
            Buffer.check_label(Tasker) #       new_hard_data=[] #all_hard_data #[]
            
            Buffer.calculate_score(working_model, Tasker, Buffer.learnable_buffer,self_update=True)
            if Buffer.hard_buffer:
                easier_data=data_generator.generate_easier_data(Buffer.hard_buffer,task_instruction,task_name,iter_num,multi_task,passage_paths)
                easy,learnable,hard=Buffer.calculate_score(working_model, Tasker,easier_data,self_update=False)
                Buffer.learnable_buffer.extend(learnable)
                Buffer.hard_buffer.extend(hard)
                Buffer.easy_buffer.extend(easy)
            if Buffer.easy_buffer:
                harder_data=data_generator.generate_hard_data(Buffer.easy_buffer,task_instruction,task_name,iter_num,multi_task,passage_paths)
                easy,learnable,hard=Buffer.calculate_score(working_model, Tasker,harder_data,self_update=False)
                Buffer.learnable_buffer.extend(learnable)
                Buffer.hard_buffer.extend(hard)
                Buffer.easy_buffer.extend(easy)
            working_model.quit()
            del working_model
            train_data=Buffer.generate_train_batch(train_num=500,prob_sampling=False,easy_to_hard=False)
            import os
            cur_path=os.getcwd()
            torch.save(train_data,os.path.join(cur_path,'src','train_data_{0}_{1}.pt'.format(task_name,iter_num))) 
            
        Tasker.process_and_save_dataset(train_data, task_name, dataset_path)
        
        RL_Trainer.dataset_path=dataset_path
        RL_Trainer.model_path=base_model_path #working_model_path #base_model_path
        RL_Trainer.experiment_name=f"1_{task_name}_{iter_num}"

        RL_Trainer.run()

        new_model_path=RL_Trainer.test_run()
        if new_model_path:
            working_model_path=new_model_path
        pdb.set_trace()
        

    print("\n--- GSM8K Training Complete ---")
    return working_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Iterative Training Framework')

    parser.add_argument('--base_model_path', type=str, required=True,
                        help='Path to the base model')
    parser.add_argument('--task_name', type=str, required=True,
                        help='Name of the task')
    parser.add_argument('--task_instruction', nargs="+", required=True,
                        help='Instructions for the task')
    parser.add_argument('--demo_examples', type=List, default=None,
                        help='Demonstration examples (can be provided multiple times)')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the dataset')
    parser.add_argument('--work_model_paths', action='append', required=True,
                        help='Paths for work models (can be provided multiple times)')
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of iterations (default: 5)')
    parser.add_argument('--passage_paths', type=List, default=['./src/retriever/passages/wiki','./src/retriever/passages/wikihow','./src/retriever/passages/stackexchange'],
                        help='Paths for retrieval passages (can be provided mannually)')
    parser.add_argument('--domain', nargs="+", default=None,
                        help='Domain keywords (default: None)')
    parser.add_argument('--multi_task', type=bool, default=False,
                        help='multi_task_setting (default: False)')
    args = parser.parse_args()
    if 'gsm8k' in args.task_name:
        args.demo_examples='[{"input":"Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?","output":"Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72"}]'
    if 'logiqa' in args.task_name:
        args.demo_examples = '[{"input":"Context:Some Cantonese don\'t like chili, so some southerners don\'t like chili. Which of the following can guarantee the above argument? Option A:Some Cantonese love chili. Option B: Some people who like peppers are southerners. Option C: All Cantonese are southerners. Option D: Some Cantonese like neither peppers nor sweets.","output":"C"}]'

        
    if 'math' in args.task_name:
        import datasets
        args.demo_examples=[]
        for name in ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']:
            dataset = datasets.load_dataset('EleutherAI/hendrycks_math', name)
            data = dataset['train'][0]
            new_data={'input':data['problem'],'output':data['solution']} 
            args.demo_examples.append([new_data])

    path=iterative_training_framework(
        base_model_path=args.base_model_path,
        task_name=args.task_name,  
        task_instruction=args.task_instruction,
        demo_examples=args.demo_examples,
        dataset_path=args.dataset_path,
        work_model_paths=args.work_model_paths,
        passage_paths=args.passage_paths,
        iterations=args.iterations,
        domain=args.domain,
        multi_task=args.multi_task
    )
