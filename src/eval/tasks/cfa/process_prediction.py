import re
def process_prediction(pred:str):
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, pred)
    matches = list(match)
    if matches:
        final_str = matches[-1].group(1).strip()
        return final_str
    return None
        
