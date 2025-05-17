import re
def process_prediction(pred:str):
        #sens=pred.split('.')
        #final_sens=[sen for sen in sens if 'final' in sen]
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, pred)
    matches = list(match)
    if matches:
        final_str = matches[-1].group(1).strip()
        option_matches=re.findall(r'\b[A-D]\b', final_str)
        if option_matches:
            return option_matches[0]
    return None
        
