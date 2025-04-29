import re
def process_label(solution_str):
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

