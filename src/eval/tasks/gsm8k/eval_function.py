from verl.utils.reward_score.gsm8k import compute_score


def eval_function(pred:str, label:str):
    eval=compute_score(pred, label,valid=True)
    if eval is None or eval==0:
        return False
    else:
        return True
        '''
        if pred is None or label is None:
            return False
        elif abs(float(pred)-float(label))>1e-3:
            return False
        else:
            return True
        '''
