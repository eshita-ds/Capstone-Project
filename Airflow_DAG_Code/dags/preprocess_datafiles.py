# Project Name: Chain of Thought Reasoning
# Team Number: 6
# Members: Anvesh, Eshita, Neha, Sandeep, Saumya

# File Name: preprocess_datafiles.py
# File Usage:
#     This file is used to preprocess the data files extracted from Amazon RDS.
#     The data files are in CSV format and are stored in S3.
#     The preprocessing includes reading the data files, generating alternative answers,
#     arranging the options, and uploading the processed data to S3 in Parquet format.
#     The processed data is then used for finetuning the model.
#     The script also archives the original data files and updates the log file in S3.
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
from io import StringIO
import openai
import time
from tqdm import tqdm
from datasets import DatasetDict, Dataset
from io import BytesIO
import pyarrow as pa
import pyarrow.parquet as pq
import os

s3 = boto3.client(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
)

bucket_name = 'finetuning-stage'
train_file_key = 'train_dataset.parquet'
valid_file_key = 'valid_dataset.parquet'
files_to_archive = []

openai.api_key = os.environ.get('OPEN_AI_KEY')
template = lambda question : f"""You are an AI tutor that thinks and provides detailed and step-by-step explanations for the provided maths question.
**Question**: {question}"""

def parse_data_files():

    prefix = ''  # or '' if listing everything in the bucket

    # List all objects with the given prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    # Initialize an empty list to collect dataframes
    all_dfs = []
    header_read = False  # flag to read header only once

    if 'Contents' in response:
        for obj in response['Contents']:
            key = obj['Key']
            if key.endswith('.csv') and 'data_from_rds' in key and '/' not in key:
                print(f"Reading: {key}")
                files_to_archive.append(key)
                file_obj = s3.get_object(Bucket=bucket_name, Key=key)
                content = file_obj['Body'].read().decode('utf-8')

                if not header_read:
                    # First file: read with header
                    df = pd.read_csv(StringIO(content))
                    columns = df.columns
                    header_read = True
                else:
                    # Subsequent files: skip header
                    df = pd.read_csv(StringIO(content), names=columns, header=None, skiprows=1)
                
                all_dfs.append(df)
    else:
        print("No matching files found.")

    # Combine all dataframes
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print("Combined DataFrame:")
    else:
        combined_df = pd.DataFrame()
    
    return combined_df

def gen_alternate_answers(df):
    df = df.copy()

    # Sample 30 rows for testing
    df_sample = df.sample(30, random_state=42).copy()

    # Function to generate a single alternative answer
    def generate_alternative_answer(question):
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI tutor that provides detailed, step-by-step explanations for math problems."},
                    {"role": "user", "content": question}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error: {e}")
            return ""

    # Generate one alternative answer per question
    alternative_answers = []
    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        alt_answer = generate_alternative_answer(row['question'])
        alternative_answers.append(alt_answer)
        time.sleep(1)

    # Add to DataFrame
    df_sample['alt_answer'] = alternative_answers
    return df_sample

def arrange_options(row):
    # Compare lengths of answer and alt_answer
    if len(row['answer']) >= len(row['alt_answer']):
        return pd.Series([row['question'], row['answer'], row['alt_answer']])
    else:
        return pd.Series([row['question'], row['alt_answer'], row['answer']])

def idea_to_prompt(question):
    return [{"role": "user", "content": template(question.lower())}]

def title_to_completion(title):
    return [{"role": "assistant", "content": title}]

def upload_to_s3(dataset, bucket_name, file_key):
    # Convert Dataset to a pyarrow Table
    table = pa.Table.from_pandas(dataset.to_pandas())
    
    # Create an in-memory buffer
    buf = BytesIO()

    # Write the table to the buffer as a Parquet file
    pq.write_table(table, buf)
    
    # Move buffer cursor to the beginning
    buf.seek(0)

    # Upload the buffer to S3
    s3.upload_fileobj(buf, bucket_name, file_key)

def archive_files():
    for file_name in files_to_archive:
        # Construct source and destination keys
        archive_prefix = 'archive/'
        source_key = f"{file_name}"  # e.g., 'data/file1.txt'
        destination_key = f"{archive_prefix}{file_name}"  # e.g., 'archive/file1.txt'
        
        try:
            # Copy file to archive directory
            s3.copy_object(
                Bucket=bucket_name,
                CopySource={'Bucket': bucket_name, 'Key': source_key},
                Key=destination_key
            )
            # Delete original file
            s3.delete_object(Bucket=bucket_name, Key=source_key)
            print(f"Moved {source_key} to {destination_key}")
        except s3.exceptions.ClientError as e:
            print(f"Error moving {source_key}: {e}")
            continue

def update_log():
    obj = s3.get_object(Bucket=bucket_name, Key='extraction-log.csv')
    log_df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    log_df.loc[log_df['file_name'].isin(files_to_archive), 'processed'] = 'Yes'
    csv_buffer = io.StringIO()
    log_df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket_name, Key='extraction-log.csv', Body=csv_buffer.getvalue())

def main():
    df = parse_data_files()
    if df.empty:
        print(f"No Files Found, Skip Further Processing!")
        return 0
    else:    
        df = df.drop_duplicates(subset='question', keep='first')
        df = df.dropna(subset=['question'])
        df = gen_alternate_answers(df)
        final_df = df[['question', 'answer', 'alt_answer']].apply(arrange_options, axis=1)
        final_df.columns = ['question', 'option1', 'option2']

        final_df['prompt'] = final_df['question'].apply(idea_to_prompt)

        final_df['chosen'] = final_df['option1'].apply(title_to_completion)
        final_df['rejected'] = final_df['option2'].apply(title_to_completion)

        df_shuffled = final_df.iloc[:,-3:].sample(frac=1, random_state=42).reset_index(drop=True)
        train_size = int(0.9*len(final_df))

        df_train = final_df.iloc[:train_size]
        df_valid = df_shuffled.iloc[train_size:]

        df_train.drop(columns=['question', 'option1', 'option2'], inplace=True)

        # Convert the pandas DataFrames back to Hugging Face Datasets
        train_ds = Dataset.from_pandas(df_train)
        valid_ds = Dataset.from_pandas(df_valid)

        # Combine into a DatasetDict
        dataset_dict = DatasetDict({
            'train': train_ds,
            'valid': valid_ds,
        })

        upload_to_s3(dataset_dict['train'], bucket_name, train_file_key)
        upload_to_s3(dataset_dict['valid'], bucket_name, valid_file_key)

        print(f"Train and Valid Parquet files uploaded to S3 successfully!")

        archive_files()

        print(f"Files Archived Successfully!")

        update_log()

        print('Log File Updated Successfully')

        return 1
