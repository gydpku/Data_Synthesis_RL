import re
import random
def extract_solution(solution_str):
    # Remove everything before the first "Assistant:"

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_str = matches[-1].group(1).strip()
        option_matches=re.findall(r'\b[A-D]\b', final_str)
        if option_matches:
            return option_matches[0]
    return None


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
    import pdb
    #pdb.set_trace()
    answer = extract_solution(solution_str=solution_str)
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
    import pdb
    #pdb.set_trace()
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
import pdb
def eval_function(pred:str, label:str):
    eval=compute_score(pred, label,valid=True)
    #pdb.set_trace()
    if eval is None or eval==0:
        return False
    else:
        return True
