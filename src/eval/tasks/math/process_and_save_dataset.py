from datasets import Dataset
import datasets
import pdb

from eval.tasks.math.process_prediction import process_prediction
from eval.tasks.math.get_output_instruction import get_output_instruction
import os

def process_and_save_dataset(train_data, data_source,local_dir,is_instruct=False):
    """
    Processes train and test datasets from lists of data lines, applies a mapping function,
    and saves them as parquet files.

    Args:
        train_data(list):training data (e.g., read from a file).
        data_source (str): Source of the data.
        local_dir (str): Local directory path to save parquet files.

    Returns:
        tuple: A tuple containing the paths to the saved train and test parquet datasets.
               Returns (train_dataset_path, test_dataset_path).
    """
    
    new_train_data=[]
    for data in train_data:
        new_data={}
        for key in data:
            try:
                new_data[key]=str(data[key])
            except:
                pdb.set_trace()
        new_train_data.append(new_data)
    train_data=new_train_data
    train_dataset=Dataset.from_list(train_data)
    ori_test_dataset = datasets.load_dataset('xDAN2099/lighteval-MATH', split='test', trust_remote_code=True)
    test_data=[]
    for data in ori_test_dataset:
        new_data={'input':data['problem'],'output':data['solution']}
        test_data.append(new_data)
    test_dataset=Dataset.from_list(test_data)
    instruction_following = get_output_instruction()

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop('input')
            if is_instruct:
                instruction= f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n"""

                instruction_following = "Let's think step by step and output the final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n"
                question = instruction + question_raw + ' ' + instruction_following
            else:
                instruction_following = get_output_instruction()
                question = question_raw + ' ' + instruction_following
            try:
                answer_raw = example.pop('output')
                solution = process_prediction(answer_raw)

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
            except Exception as e:
                print(f"Error processing example at index {idx} in split {split}: {e}")
                return None  # Or handle the error as appropriate, e.g., return None and filter later

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=['input', 'output']) #remove original columns
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, remove_columns=['input', 'output']) #remove original columns

    train_dataset_path = os.path.join(local_dir, 'train.parquet')
    test_dataset_path = os.path.join(local_dir, 'test.parquet')

    train_dataset.to_parquet(train_dataset_path)
    test_dataset.to_parquet(test_dataset_path)
