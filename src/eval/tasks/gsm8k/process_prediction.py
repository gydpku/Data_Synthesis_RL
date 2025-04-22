import re
from verl.utils.reward_score.gsm8k import extract_solution
def process_prediction(pred:str):
    answer=extract_solution(pred,method='strict')
    return answer
    """
        predicted_output=pred
        if '####' in predicted_output:
            predicted_output=predicted_output[predicted_output.find('####'):]
        elif '###' in predicted_output:
            predicted_output=predicted_output[predicted_output.find('###'):]
        elif '##' in predicted_output:
            predicted_output=predicted_output[predicted_output.find('##'):]
            
        regex_pattern = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
        regexes_to_ignore =[
            ",",
            "\\$",
            "(?s).*#### ",
            "\\.$"
        ]
        match = re.findall(regex_pattern, predicted_output)
        if match:
            print(predicted_output,'\n')
            print(match,'\n')
            print(match[-1],'\n')
            match = match[-1]
            if isinstance(match, tuple):
                match = [m for m in match if m][0]
            text = match.strip()
    
            for regex in regexes_to_ignore:
                text = re.sub(regex, "", text)
    #        print(text,predicted_output)
            if text.count('.') > 1:
            # Retain only up to the last valid dot
                index_1=text.find('.')
                index_2=text.find('.',index_1+1)
                text = text[:index_2]
            try:
                return float(text)
            except ValueError as e:
                print('Error:', e)
                return None
        else:
            return None
        return pred
    """
