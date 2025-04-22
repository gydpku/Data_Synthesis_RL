def process_label(label:str) -> Optional[str]:
        if 'neutral' in label.lower():
                return 'neutral'
            if 'contradiction' in label.lower():
                return 'contradiction'
            if 'entailment' in label.lower():
                return 'entailment'
        return None
