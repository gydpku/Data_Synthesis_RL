import re
def process_prediction(pred:str) -> Optional[str]:
    predicted_sql,db_path = pred[0],pred[1]
        
    prior_pred=predicted_sql.split('final SQL')[0]
    try:
        predicted_sql = predicted_sql.split('final SQL')[1].strip()
    except:
        predicted_sql = 'SELECT'+predicted_sql.split('SELECT')[1]
    predicted_sql=predicted_sql.split(';')[0]
    predicted_sql=predicted_sql[predicted_sql.find('SELECT'):] #[1:]
    conn=sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        return predicted_res
    except:
        return None
    return None
