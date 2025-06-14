# Project Name: Chain of Thought Reasoning
# Team Number: 6
# Members: Anvesh, Eshita, Neha, Sandeep, Saumya

# File Name: data_extract.py
# File Usage: 
#     This file is used to extract data from Amazon RDS and push it to S3.
#     The data is extracted from the user_chats table in the RDS database.
#     The data is then pushed to S3 in CSV format.
#     The log file is also updated in S3 to keep track of the extraction process.
#     The log file contains the date of extraction, source of data, number of records extracted,
#     whether the data has been processed, and the file name.
#     The log file is used to keep track of the extraction process and to avoid duplicate extractions.
#     The log file is stored in S3 and is updated after each extraction.

import boto3
import pandas as pd
import io
from datetime import datetime, timedelta
import ast
import sys
from sqlalchemy import create_engine, text
import os

# =============== Configuration ===============

# S3 setup with credentials (only needed if not configured globally)
s3 = boto3.client(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
)

bucket_name = 'finetuning-stage'
log_file_key = 'extraction-log.csv'         

# RDS Credentials
db_user = os.environ.get('MYSQL_DB_USER')
db_password = os.environ.get('MYSQL_DB_PASSWORD')
db_endpoint = os.environ.get('MYSQL_DB_ENDPOINT')
db_port = os.environ.get('MYSQL_DB_PORT')
db_name = os.environ.get('MYSQL_DB_NAME')


# Step 1: Read log file from S3
def get_last_extract_date():
        log_obj = s3.get_object(Bucket=bucket_name, Key=log_file_key)
        log_df = pd.read_csv(io.BytesIO(log_obj['Body'].read()))

        # Parse 'date' column as datetime
        log_df['date'] = pd.to_datetime(log_df['date'])

        if not log_df['date'].dropna().empty:
            last_extracted_date = log_df['date'].max()
            # print(last_extracted_date)
        else:
            last_extracted_date = datetime(2000, 1, 1)

        # Step 2: Get latest extraction date
        # last_extracted_date = log_df['date'].max()
        return last_extracted_date

def extract_rds_records(last_extracted_date):
    # Create the connection engine
    engine = create_engine(f'mysql+pymysql://{db_user}:{db_password}@{db_endpoint}:{db_port}/{db_name}')
    # Test the connection
    try:
        connection = engine.connect()
        # print("Connected successfully to the database!")
    except Exception as e:
        print("Connection failed:", e)

    today = datetime.now().date()

    # Ensure last_extracted_date is in datetime format
    last_extract_date = last_extracted_date.strftime('%Y-%m-%d %H:%M:%S')

    query = text("SELECT * FROM user_chats WHERE create_time > :last_date")
    df = pd.read_sql(query, engine, params={"last_date": last_extract_date})

    # Safely parse 'history' strings into real Python lists
    df['parsed_history'] = df['history'].apply(lambda x: ast.literal_eval(x))

    # Create a list to store the question-answer pairs
    qa_pairs = []

    for history in df['parsed_history']:
        for question, answer in history:
            qa_pairs.append({
                'question': question,
                'answer': answer
            })

    # Create the final DataFrame
    qa_df = pd.DataFrame(qa_pairs)

    return qa_df

def push_to_s3(s3, df, file_name, number_of_records):
    # Push file to S3
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    s3.put_object(Bucket=bucket_name, Key=file_name, Body=csv_buffer.getvalue())
    # s3.upload_file(file_name, 'finetuning-stage', file_name) 
    # source_file_key = file_name 

    # Update Log File
    obj = s3.get_object(Bucket=bucket_name, Key=log_file_key)
    log_df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    log_columns = ['date', 'source', 'number_of_records', 'processed', 'file_name']

    if log_df.empty:
        log_df = pd.DataFrame(columns=log_columns)
    else:
        log_df = log_df[log_columns] if all(col in log_df.columns for col in log_columns) else log_df

    new_entry = pd.DataFrame([{
        'date': datetime.now().strftime('%Y-%m-%d'),
        'source': "RDS",
        'number_of_records': number_of_records,
        'processed': 'No',
        'file_name' : file_name
    }])
    log_df = pd.concat([log_df, new_entry], ignore_index=True)

    # --- Write updated log back to S3 ---
    log_buffer = io.StringIO()
    log_df.to_csv(log_buffer, index=False)

    s3.put_object(Bucket=bucket_name, Key=log_file_key, Body=log_buffer.getvalue())

def main():
    last_extracted_date = get_last_extract_date()
    df = extract_rds_records(last_extracted_date)
    number_of_records = len(df)

    if number_of_records > 0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        file_name = f"data_from_rds_{timestamp}.csv"
        # df.to_csv(file_name, index = False)
        push_to_s3(s3, df, file_name, number_of_records)
        print(f"number of records: {number_of_records}")
        return number_of_records
    else:
        print("There are no records in the dataframe")  
        return 0
