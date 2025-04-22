def eval_function(pred:str, label:str):
        if pred is None or set(pred) != set(label)::
            return False
        else: 
            return True
