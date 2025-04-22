import re
from eval.tasks.gsm8k.process_prediction import process_prediction
def process_label(label:str):
    try_1=process_prediction(label)
    if try_1 is not None:
        return try_1
    else:
        if len(label.split('\n\n'))>1:
          return process_prediction(label.split('\n\n')[-2])
        else:
          return
    return label
