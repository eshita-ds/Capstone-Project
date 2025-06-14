# Project Name: Chain of Thought Reasoning
# Team Number: 6
# Members: Anvesh, Eshita, Neha, Sandeep, Saumya

# File Name: cot_llm_ft_pipeline.py
# File Usage: 
#     This file is used to create an Airflow DAG for the Chain of Thought Reasoning project.
#     The DAG consists of several tasks that are executed in a specific order.
#     The tasks include extracting data from Amazon RDS, preprocessing the data, finetuning the model,
#     submitting evaluation for the new model, comparing metrics, and deciding the final output.
#     The DAG is scheduled to run daily and does not catch up on past runs.
#     The DAG uses PythonOperator, BranchPythonOperator, and BashOperator to execute the tasks.
#     The DAG also uses XCom to pass data between tasks.
#     The DAG is defined using the Airflow DAG class and is executed in a specific order.



from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.decorators import task
from datetime import datetime
from random import randint
import data_extract as data_extract
import preprocess_datafiles as ppd
import submit_finetuning as sf
import submit_ft_eval as sfe
import compare_metrics as cm

# Extract Data from Amazon RDS
def _run_rds_extract():
    return data_extract.main()

# Decide next move after data extraction
def _decide_next_move(ti=None):
    # number_of_records = ti.xcom_pull(task_ids = [
    #     'rds_database_extract'
    # ])
    number_of_records = ti.xcom_pull(task_ids = 'rds_database_extract')
    print(number_of_records)

    if number_of_records > 0:
         return "proceed"
    return "stop"

# Preprocess Extracted Data
def _run_preprocess_data():
    return ppd.main()

# Decide next move after data preprocessing
def _ready_for_tuning(ti=None):
    files_available = ti.xcom_pull(task_ids = 'preprocess_data')

    if files_available == 1:
         return "Yes"
    return "No"

# Finetune the Model using Preprocessed Data
def _run_finetune_model():
    return sf.main()

# Submit Evaluation for New Model
def _submit_evaluation():
    return sfe.main()

# Compare Final Metrics
def _compare_metrics():
    return cm.main()

# Decide Final Output
def _decide_result(ti=None):
    improved_y_n = ti.xcom_pull(task_ids = 'compare_metrics')
    print(improved_y_n)

    if improved_y_n == "Y":
        return "model_version_updated"
    return "model_version_discarded"

# Start DAG
with DAG("cot_llm_ft_pipeline", start_date=datetime(2025, 4, 27), schedule="@daily", catchup=False) as dag:
    
    rds_database_extract = PythonOperator(
        task_id='rds_database_extract',
        python_callable=_run_rds_extract,
        do_xcom_push=True
    )

    decide_next_move = BranchPythonOperator(
        task_id = "decide_next_move",
        python_callable = _decide_next_move
    )

    proceed = BashOperator(
        task_id = "proceed",
        bash_command = "echo 'proceed'"
    )

    stop = BashOperator(
        task_id = "stop",
        bash_command = "echo 'stop'"
    )

    preprocess_data = PythonOperator(
        task_id='preprocess_data',
        python_callable=_run_preprocess_data,
        do_xcom_push=True
    )

    ready_for_tuning = BranchPythonOperator(
        task_id = "ready_for_tuning",
        python_callable = _ready_for_tuning
    )

    yes = BashOperator(
        task_id = "Yes",
        bash_command = "echo 'Yes'"
    )

    no = BashOperator(
        task_id = "No",
        bash_command = "echo 'No'"
    )

    fine_tune_model = PythonOperator(
        task_id='fine_tune_model',
        python_callable=_run_finetune_model,
        do_xcom_push=True
    )

    submit_evaluation = PythonOperator(
        task_id='submit_evaluation',
        python_callable=_submit_evaluation,
        do_xcom_push=True
    )

    compare_metrics = PythonOperator(
        task_id='compare_metrics',
        python_callable=_compare_metrics,
        do_xcom_push=True
    )

    decide_result = BranchPythonOperator(
        task_id='decide_result',
        python_callable = _decide_result
    )

    model_version_updated = BashOperator(
        task_id = "model_version_updated",
        bash_command = "echo 'Model Version Updated'"
    )

    model_version_discarded = BashOperator(
        task_id = "model_version_discarded",
        bash_command = "echo 'Model Version Discarded'"
    )

    rds_database_extract >> decide_next_move >> [proceed, stop]
    proceed >> preprocess_data >> ready_for_tuning >> [yes, no]
    yes >> fine_tune_model >> submit_evaluation >> compare_metrics >> decide_result >> [model_version_updated, model_version_discarded]