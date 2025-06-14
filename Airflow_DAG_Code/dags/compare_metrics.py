# Project Name: Chain of Thought Reasoning
# Team Number: 6
# Members: Anvesh, Eshita, Neha, Sandeep, Saumya

# File Name: compare_metrics.py
# File Usage:
#     This file is used to compare the latest g_eval_score with the previous one.
#     If the latest g_eval_score is greater than the previous one, it updates the database with the new score and revision.
#     If the latest g_eval_score is not greater, it does not update the database.

import mysql.connector
import os

# RDS Credentials
db_user = os.environ.get('MYSQL_DB_USER')
db_password = os.environ.get('MYSQL_DB_PASSWORD')
db_endpoint = os.environ.get('MYSQL_DB_ENDPOINT')
db_port = os.environ.get('MYSQL_DB_PORT')
db_name = os.environ.get('MYSQL_DB_NAME')

DB_CONFIG = {
    "host": db_endpoint,
    "port": db_port,
    "user": db_user,
    "password": db_password,
    "database": db_name,
}

def main():
    model_name = "Eshita-ds/Llama-3.2-1B-DPO"
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(buffered=True)

    cursor.execute("""SELECT g_eval_score, revision, previous_revision, latest_commit, latest_geval
                    FROM cot_models where model_name = %s""", (model_name,))
    sessions = cursor.fetchall()
    conn.close()

    g_eval_score = sessions[0][0]
    revision = sessions[0][1]
    previous_revision = sessions[0][2]
    latest_commit = sessions[0][3]
    latest_geval = sessions[0][4]

    print(f'{g_eval_score}-{revision}-{previous_revision}-{latest_commit}-{latest_geval}')

    if latest_geval > g_eval_score:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(buffered=True)
        cursor.execute("""UPDATE cot_models SET g_eval_score = %s, revision = %s, previous_revision = %s
                        WHERE model_name = %s""", 
                (latest_geval, latest_commit, revision, model_name))
        conn.commit()
        conn.close()
        return "Y"
    else:
        return "N"