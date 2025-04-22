def process_prediction(pred:str) -> Optional[str]:
        sens=pred.split('.')
        final_sens=[sen for sen in sens if 'final' in sen]
        for text in final_sens:
            if 'neutral' in text.lower():
                return 'neutral'
            if 'contradiction' in text.lower():
                return 'contradiction'
            if 'entailment' in text.lower():
                return 'entailment'
        return None
