import re
def process_label(label:str) -> Optional[str]:
    ground_truth,db_path=label[0],label[1]
    conn=sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(ground_truth)
        ground_truth_res = cursor.fetchall()
        return ground_truth_res
    except:
        return 
        
