from verl.utils.reward_score.medqa import compute_score
def eval_function(pred:str, label:str):
    eval=compute_score(pred, label,valid=True)
    if eval is None or eval==0:
        return False
    else:
        return True
