# Project Name: Chain of Thought Reasoning
# Team Number: 6
# Members: Anvesh, Eshita, Neha, Sandeep, Saumya

# File Name: submit_finetuning.py
# File Usage:
#     This file is used to submit the finetuning job to Kaggle.
#     The finetuning job is submitted using the Kaggle API.
#     The job is submitted in the form of a kernel.
#     The kernel is created using the Kaggle API and is pushed to Kaggle.
#     The kernel is then run on Kaggle and the status is checked.
#     The status is checked using the Kaggle API.
#     The status is checked every 30 seconds until the job is completed or errors out.
#     The status is printed to the console.


from kaggle import KaggleApi
import os
import json
import subprocess
import time
import re

def extract_status(output):
    match = re.search(r'status\s+"KernelWorkerStatus\.(\w+)"', output)
    if match:
        return match.group(1).lower()
    return None

def wait_for_kaggle_kernel(kernel_slug, poll_interval=30):
    """
    Poll Kaggle kernel status until it completes or errors.
    
    Args:
        kernel_slug (str): Format 'username/kernel-slug'
        poll_interval (int): Seconds to wait between status checks
    """
    while True:
        result = subprocess.run(
            ["kaggle", "kernels", "status", kernel_slug],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ}
        )
        output = result.stdout.strip()
        print(output)  # optional: log status
        
        # Parse status from output
        # if '"complete"' in output.lower() or '"error"' in output.lower():
        #     print("Kernel is no longer running.")
        #     break
        status = extract_status(output)
        if status in {"complete", "error"}:
            print(f"Kernel finished with status: {status}")
            break
        
        time.sleep(poll_interval)

def main():
    notebook_folder = "/opt/airflow/dags/kaggle-ft"
    result = subprocess.run(
        ["kaggle", "kernels", "push", "-p", notebook_folder],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ}
    )
    print(result.stdout)

    wait_for_kaggle_kernel("eshitagupta151991/airflow-llama-ft-v2", 30)

    print("Process Finished")
