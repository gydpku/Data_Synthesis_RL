import re
import random
import datasets

def extract_solution(solution_str):
    # Remove everything before the first "Assistant:"
#    solution_str=solution_str[solution_str.find('</answer>')+len('</answer>'):]
    answer_pattern = r'<answer>(.*?)</answer>'
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
    dataset = datasets.load_from_disk('/dccstor/obsidian_llm/yiduo/summary/src/medical_qa')
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    train_data=[]
    test_data=[]
    def get_gold(data):
            try:
                gold=data['std']
            except:
                gold=data['answer']
            return gold
    instruction = 'Your output thinking process and answer should be enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> thinking process here </think> <answer> the correct option here </answer>. '
    for data in train_dataset:
        train_data.append({'input':data['query'][data['query'].find('.')+1:].strip()+' '+instruction,'answer':get_gold(data)})
    for data in test_dataset:
        test_data.append({'input':'Context: '+' '.join(data['rag'])+' '+data['query'][data['query'].find('.')+1:].strip()+' '+instruction,'answer':get_gold(data)})
    return train_data,test_data
def compute_score(solution_str, ground_truth, method='strict',valid=False, format_score=0.1, score=1.):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    if do_print:
        print(f"--------------------------------")
        print(f"Ground truth: {ground_truth} | Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
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
