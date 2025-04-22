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

import re
import pdb
import datasets

def extract_solution(solution_str, method='strict'):
    assert method in ['strict', 'flexible']

    if method == 'strict':
        # this also tests the formatting of the model
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = final_answer.split('#### ')[1].replace(',', '').replace('$', '')
    elif method == 'flexible':
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ['', '.']
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer
def load_data():
    dataset = datasets.load_dataset('openai/gsm8k', 'main')

    train_dataset = dataset['train']
    test_dataset = dataset['test']
    train_data=[]
    test_data=[]
    instruction_following = "Let's think step by step and output the final answer after \"####\"."
    for data in train_dataset:
        train_data.append({'input':data.pop('question')+' '+instruction_following,'answer':extract_solution(data.pop('answer'))})
    for data in test_dataset:
        test_data.append({'input':data.pop('question')+' '+instruction_following,'answer':extract_solution(data.pop('answer'))})
    return train_data,test_data    
def compute_score(solution_str, ground_truth, method='strict',valid=False, format_score=0., score=1.):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    from R1_tuner.src.eval.tasks.gsm8k.process_prediction import process_prediction
    from R1_tuner.src.eval.tasks.gsm8k.eval_function import eval_function
    pred=process_prediction(solution_str)
    if pred is None:
        return 0
    else:
        if eval_function(pred, ground_truth):
            return score
        else:
            return format_score if not valid else 0
    
