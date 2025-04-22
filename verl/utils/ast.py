import time
import pdb
import json
import re
import random
import datasets
from bfcl.eval_checker.ast_eval.ast_checker import ast_checker

def extract_solution(solution_str):
    # Remove everything before the first "Assistant:"

    answer_pattern = r'<code>(.*?)</code>'
    #print("solution_str",solution_str)
    #import time
    #time.sleep(100)
    function=''
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
#    if final_answer is not None:
 #       try:
 #           int_final_answer = int(final_answer)
 #       except ValueError:
 #           final_answer = None
    return final_answer

def load_data():
    dataset = datasets.load_dataset('lucasmccabe/logiqa', trust_remote_code=True)

    train_dataset = dataset['train'].select(range(100))
    val_dataset = dataset['validation']
    test_dataset = dataset['test']
    train_data=[]
    test_data=[]
    instruction = 'Your output thinking process and answer should be enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> thinking process here </think> <answer> the correct option here </answer>. '
    for data in train_dataset:
        train_data.append({'input':data['context']+' '+data['query']+ ' '+' '.join([str(chr(id+ord('A')))+': '+option for id,option in enumerate(data['options'])])+' '+instruction,'answer':chr(int(data['correct_option'])+ord('A'))})
    for data in test_dataset:
        test_data.append({'input':data['context']+' '+data['query']+ ' '+' '.join([str(chr(id+ord('A')))+': '+option for id,option in enumerate(data['options'])])+' '+instruction,'answer':chr(int(data['correct_option'])+ord('A'))})
    return train_data,test_data
def find_index(string):
        pattern = r'for index (\d+) of possible answers'

# Loop through the example texts and extract the number
        match = re.search(pattern, string)
        if match:
            extracted_index = match.group(1)  # The number after 'for index'
            return extracted_index #print(f"Extracted index: {extracted_index}")
        else:
            return 0 #print("No match found.")

def compute_score(solution_str, ground_truth, method='strict', valid=False, format_score=0.1, score=1.):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
 #   print("solution_str",solution_str)
    function_raw=solution_str[solution_str.find('Current Question:'):]
    function_raw=function_raw[:function_raw.find('Model response:')]
    function_raw=function_raw[function_raw.find('Here is a list of functions'):]
    function_raw=function_raw[function_raw.find('['):function_raw.rfind(']')+1]
    import pdb
#    pdb.set_trace()
 #   time.sleep(100)
    answer = extract_solution(solution_str=solution_str)
#    print("solution_str",solution_str)
 #   time.sleep(100)
  #  print("answer",answer)
    ground_truth=str(ground_truth) #  print("func",eval(function_raw))
    try:
        ground_truth=eval(ground_truth)
    except:
        ground_truth=json.loads(ground_truth)
  #  print("ground",ground_truth)
    language="Python"
    try:
        answer=answer[answer.find('['):answer.rfind(']')+1]
        pred=eval(answer)

    except:
        return -1 if not valid else 0
    if not isinstance(pred,list) or isinstance(pred,int) or not isinstance(ground_truth,list):
        return -1 if not valid else 0
#    if not isinstance(pred[0],dict):
 #       return -1 if not valid else 0
    def single_reward(delta,error_type):
        if 'wrong_func_name' in error_type:
            return delta*1/7 
        elif 'missing' in error_type:
            return delta*2/7 
        elif 'unexpected' in error_type: 
            return delta*3/7 
        elif 'type_error' in error_type: 
            return delta*4/7  
        elif 'value_error' in error_type: 
            return delta*5/7 
        elif 'missing_optional' in error_type: 
            return delta*6/7 
        return 0
    model_name="ooo"
  #  print("label",ground_truth)
#    time.sleep(10)
    try:
        function=eval(function_raw)
    except:
        try:
            function=json.loads(function_raw)
        except:
            print("func",function_raw)
    if len(function)>1:
        result=ast_checker(function,pred,ground_truth,language,'multiple',model_name)
        print('result_mul',result)
#        pdb.set_trace()
    elif len(ground_truth)>1:
        result=ast_checker(function,pred,ground_truth,language,'parallel',model_name)
        print('result_parallel',result)
        if result['valid']:
            return 1
        elif valid:
            return 0
        elif 'Wrong number' in result['error'][0]:        
            return -1
        delta=2/len(ground_truth)
        all_errors=result['error']
        index=find_index(all_errors[0]) #[all_errors[0].find('model output for index'):]
        correct_f_reward=int(index)*delta
        wrong_f_rewards=[]
        for item in all_errors[1:]:
            for sub_item in item:
                print("item",sub_item,item)
                wrong_f_rewards.append(single_reward(delta,item[sub_item]["sub_error_type"]))
        print('reward',correct_f_reward,wrong_f_rewards)
        #pdb.set_trace()
        return correct_f_reward+max(wrong_f_rewards)-1
    else:
        result=ast_checker(function,pred,ground_truth,language,'single',model_name)
#    print('result',result)
#    pdb.set_trace()
    if result['valid']:
        return 1
    elif 'Wrong number' in result['error'][0]:
        return 1-0.9 if not valid else 0
    elif not result['error']:
        return 1-0.8 if not valid else 0
    elif 'not found in model' in result['error'][0]:
        return 1-0.7 if not valid else 0
    elif 'Missing required parameter' in result['error'][0]:
        return 1-0.6 if not valid else 0
    elif 'Unexpected parameter' in result['error'][0]: 
        return 1-0.5 if not valid else 0
    elif 'Incorrect type for parameter' in result['error'][0]: 
        return 1-0.4 if not valid else 0 
    elif 'Invalid value' in result['error'][0]: 
        return 1-0.2 if not valid else 0
    elif 'Optional parameter' in result['error'][0]: 
        return 1-0.1 if not valid else 0
    return 0
    do_print = random.randint(1, 64) == 1
    if do_print:
        print(f"--------------------------------")
        print(f"Ground truth: {ground_truth} | Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
#    answer_pattern = r'<answer>(.*?)</answer>'
 #   match = re.finditer(answer_pattern, solution_str)
  #  matches = list(match)
    #if matches:
    #    final_answer = matches[-1].group(1).strip()
   # print(matches)
   # import time
   # time.sleep(120)
   # print('answer!!!',match,'s!',matches[-1],'g!',matches[-1].group(1),answer,ground_truth,'\n')
    if answer is None:
        if do_print:
            print(f"No answer found")
        return 0
    else:
        if answer == ground_truth:
            if do_print:
                print(f"Correct answer: {answer}")
            return score
        else:
            if do_print:
                print(f"Incorrect answer {answer} | Ground truth: {ground_truth}")
            return format_score if not valid else 0
