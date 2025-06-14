# Project Name: Chain of Thought Reasoning
# Team Number: 6
# Members: Anvesh, Eshita, Neha, Sandeep, Saumya

# File Name: app.py
# File Usage: 
# This file is used to create a Gradio app for an Intelligent Tutoring System.
# This is hosted on Hugging Face Spaces and uses Hugging Face Zero GPU for Inference.
# It connects to Amazon RDS database for user management and chat history.
# It includes user registration, login, chat history management, and model selection for generating responses.
# It also includes usage statistics and visualizations for user interactions.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 22:04:11 2025

@author: eshitashitij
"""

import gradio as gr
import mysql.connector
import hashlib
import secrets
import os
import re
import json
import random
import string
# import datetime
from huggingface_hub import login, list_models
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from transformers.modeling_utils import init_empty_weights
from mysql.connector import Error
from huggingface_hub import snapshot_download
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import shutil

plt.rcParams['font.family'] = 'Arial'  # Set font to Arial
plt.rcParams['font.size'] = 8

# custom_css = os.path.join(os.getcwd(), 'styles.css')
custom_css=""
with open("styles.css", "r") as file:
    custom_css = file.read()

# MySQL Database Configuration
# Using environment variables for security
db_user_name = os.getenv("RDS_DB_USER")
db_password = os.getenv("RDS_DB_PASS")
db_host = os.getenv("RDS_DB_HOST")
db_port = os.getenv("RDS_DB_PORT")
db_name = os.getenv("RDS_DB_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
print("HF_TOKEN", flush=True)
login(token=HF_TOKEN)


# MySQL Database Configuration
# Using environment variables for security
DB_CONFIG = {
    "host": db_host,
    "port": int(db_port),
    "user": db_user_name,
    "password": db_password,
    "database": db_name,
}

from huggingface_hub import snapshot_download
import os

USERNAME = "Eshita-ds"

# def download_models():
#     base_path = "./models"
#     os.makedirs(base_path, exist_ok=True)

#     # Get list of repos under your HF namespace
#     user_models = list_models(author=USERNAME)
#     print(user_models)

#     for repo in user_models:
#         repo_id = repo.modelId  # e.g., "your-username/model-A"
#         model_name = repo_id.split("/")[-1]  # just "model-A"
#         local_dir = os.path.join(base_path, model_name)

#         if not os.path.exists(local_dir) or not os.listdir(local_dir):
#             print(f"Downloading {repo_id} to {local_dir}...", flush=True)
#             snapshot_download(
#                 repo_id=repo_id,
#                 local_dir=local_dir,
#                 local_dir_use_symlinks=False
#             )
#         else:
#             print(f"{model_name} already exists. Skipping download.", flush=True)

def download_models():
    base_path = "./models"
    if os.path.exists(base_path):
        print(f"Cleaning up existing directory: {base_path}", flush=True)
        shutil.rmtree(base_path)

    os.makedirs(base_path, exist_ok=True)
    print(f"Created directory: {base_path}", flush=True)

    # Fetch model names and revisions from MySQL
    model_list = []
    try:
        conn = mysql.connector.connect(**DB_CONFIG)

        if conn.is_connected():
            cursor = conn.cursor()
            query = "SELECT model_name, revision FROM cot_models"
            cursor.execute(query)
            model_list = cursor.fetchall()  # List of tuples: [(model_name, revision), ...]
            
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return
    
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

    # Download each model with its specific revision
    for model_name, revision in model_list:
        repo_id = model_name  # e.g., "Eshita-ds/Llama-3.2-1B-DPO"
        model_name_only = model_name.split("/")[-1]  # e.g., "Llama-3.2-1B-DPO"
        local_dir = os.path.join(base_path, model_name_only)

        # Check if the model with the specific revision is already downloaded
        revision_file = os.path.join(local_dir, ".gitattributes")  # Example file to check existence
        revision_exists = os.path.exists(revision_file)  # Basic check for model presence

        if not revision_exists:
            print(f"Downloading {repo_id} (revision: {revision}) to {local_dir}...", flush=True)
            snapshot_download(
                repo_id=repo_id,
                revision=revision,  # Specify the exact revision
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
        else:
            print(f"{model_name_only} (revision: {revision}) already exists. Skipping download.", flush=True)

download_models()

def generate_session_id():
    # Define the character set: letters (A-Z, a-z) and digits (0-9)
    characters = string.ascii_letters + string.digits  # "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    # Generate an 8-character session ID
    session_id = ''.join(random.choices(characters, k=8))
    return session_id


# Database setup
def init_db():
    # First connect without specifying database to create it if it doesn't exist
    conn = mysql.connector.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
    )
    cursor = conn.cursor()

    # Create database if it doesn't exist
    # cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
    cursor.execute(f"USE {DB_CONFIG['database']}")

    # Create User table
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS User (
        user_id INT AUTO_INCREMENT PRIMARY KEY,
        first_name VARCHAR(255) NOT NULL,
        last_name VARCHAR(255) NOT NULL,
        email VARCHAR(255) UNIQUE NOT NULL,
        password VARCHAR(255) NOT NULL
    )
    """
    )
    conn.commit()
    conn.close()

from huggingface_hub import InferenceClient
client = InferenceClient(
    provider="sambanova",
    token=HF_TOKEN,
)

@spaces.GPU()
def generate_response(model_name, question):
    print(f"Model Name: {model_name}", flush=True)
    print(f"question: {question}", flush=True)
    if model_name == "Llama-3.2-1B-DPO (Best)":
        model_name = "Llama-3.2-1B-DPO"
    MODEL_NAME = model_name
    BASE_PATH = "./models"
    MODEL_PATH = os.path.join(BASE_PATH, MODEL_NAME)

    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    question = question
    prompt = f"Question: {question}\nPlease provide a step-by-step solution:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1000, temperature=0.7, do_sample=True)
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"result: {result}", flush=True)
    return result

#@spaces.GPU()
def ask_distilgpt2(question):
    # Create the chat completion with streaming
    output = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
        stream=True,
        max_tokens=1024,
    )

    # Variable to store the complete response
    response = ""

    # Iterate over the streaming output and build the response
    for chunk in output:
        # Extract the content from each chunk
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta.content:  # Check if there's content in the delta
                response += delta.content

    # Return the complete response
    return response

## NEW PASTED
def get_past_conversations():
    #print(f"UserName: {session.user}", flush=True)
    if session.user != None:
        #print(f"UserName: {session.user["user_id"]}", flush=True)
        #print(f"Session ID: {session.user["session_id"]}", flush=True)
        #print(f"Inside User Not Empty", flush=True)
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(buffered=True)
        
        cursor.execute("""SELECT ROW_NUMBER() OVER (ORDER BY id) AS sequence_num, id, session_id, SUBSTRING_INDEX(SUBSTRING_INDEX(history, '",', 1), '["', -1) AS extracted_question 
                       FROM user_chats where user_id = %s""", (session.user["user_id"],))
        sessions = cursor.fetchall()
        conn.close()
        # return {str(s[0]): f"Conversation {s[0]}" for s in sessions}
        # return {str(s[1]): f"Topic: {s[3]} (ID: {s[2]})" for s in sessions}
        return {
            str(s[1]): f"{s[3].strip() if s[3] and s[3].strip() not in ('[', '[]', '') else 'New Chat'} (ID: {s[2]})"
            for s in sessions
        }
    else:
        #print(f"Inside User Empty", flush=True)
        return {}

# def get_model_names():
#     user_models = list_models(author="Eshita-ds")
    
#     model_dict = {}
#     for i, model in enumerate(user_models, start=1):
#         # Split the full model ID (e.g., Eshita-ds/phi-2-DPO)
#         _, model_name = model.modelId.split("/")
#         # model_dict[str(i)] = model_name  # Key as string for compatibility
#         model_dict[str(i)] = f"{model_name}"
#         # print(model_dict, flush=True)
#     return model_dict

def get_model_names():
    # user_models = list_models(author="Eshita-ds")
    
    model_dict = {}
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(buffered=True)
    
    # Query to fetch model_name from cot_models table
    query = "SELECT model_name FROM cot_models"
    cursor.execute(query)
    
    # Fetch all model names
    model_names = cursor.fetchall()
    
    # Build model_dict
    for i, model_name_tuple in enumerate(model_names, start=1):
        full_model_name = model_name_tuple[0]  # Extract model_name from tuple
        model_name = full_model_name.split("/")[-1]
        if model_name == "Llama-3.2-1B-DPO":
            model_name = model_name+" (Best)"
        model_dict[str(i)] = model_name
    
    cursor.close()
    conn.close()

    return model_dict

# Loading chat history
def load_chat(session_id):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(buffered=True)
    
    cursor.execute("SELECT history FROM user_chats WHERE user_id=%s and session_id=%s", 
                   (session.user["user_id"],session_id))
    history = cursor.fetchone()
    conn.close()
    
    if history:
        try:
            loaded_history = json.loads(history[0])
            if isinstance(loaded_history, list) and all(isinstance(item, list) and len(item) == 2 for item in loaded_history):
                return loaded_history
        except json.JSONDecodeError:
            return []
    return []


def save_chat(session_id, messages):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(buffered=True)
    
    cursor.execute("UPDATE user_chats SET history = %s WHERE user_id = %s and session_id = %s", 
                   (json.dumps(messages), session.user["user_id"], session.user["session_id"]))
    conn.commit()
    conn.close()

# Chatbot response
def chatbot_response(user_input, history, session_id, model_name):
    if history is None:
        history = []

    if not isinstance(history, list):
        history = []

    if isinstance(history, list) and all(isinstance(item, list) and len(item) == 2 for item in history):
        pass
    else:
        history = []

    # bot_response = ask_distilgpt2(user_input) # "Here will be the response from trained model"
    bot_response = generate_response(model_name, user_input)
    new_entry = [user_input, bot_response]
    history.append(new_entry)
    
    save_chat(session_id, history)

    return history, history, ""

## NEW PASTER OVER

# Hash password
def hash_password(password):
    salt = secrets.token_hex(16)
    pw_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}${pw_hash}"


# Verify password
def verify_password(stored_password, provided_password):
    salt, stored_hash = stored_password.split("$")
    computed_hash = hashlib.sha256((provided_password + salt).encode()).hexdigest()
    return computed_hash == stored_hash


# User registration
def register_user(first_name, last_name, email, password):
    # Input validation
    if not all([first_name, last_name, email, password]):
        return False, "All fields are required"

    # Email validation
    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if not re.match(email_pattern, email):
        return False, "Invalid email format"

    # Password length validation
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(buffered=True)

        # Check if email already exists
        cursor.execute("SELECT email FROM User WHERE email = %s", (email,))
        if cursor.fetchone():
            conn.close()
            return False, "Email already registered"

        # Hash the password
        hashed_password = hash_password(password)

        # Insert new user
        cursor.execute(
            "INSERT INTO User (first_name, last_name, email, password) VALUES (%s, %s, %s, %s)",
            (first_name, last_name, email, hashed_password),
        )
        conn.commit()
        conn.close()
        return True, "Registration successful"
    except Exception as e:
        return False, f"Error during registration: {str(e)}"


# User login
def login_user(email, password):
    new_session_id = generate_session_id()
    if not email or not password:
        return False, "Email and password are required"

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(buffered=True)
        cursor.execute(
            "SELECT user_id, first_name, last_name, password FROM User WHERE email = %s",
            (email,),
        )
        user = cursor.fetchone()
        conn.close()

        if not user:
            return False, "Invalid email or password"

        user_id, first_name, last_name, stored_password = user

        if verify_password(stored_password, password):
            return True, {
                "user_id": user_id,
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "session_id": new_session_id
            }
        else:
            return False, "Invalid email or password"
    except Exception as e:
        return False, f"Login error: {str(e)}"


# Session state to track logged-in user
class SessionState:
    def __init__(self):
        self.user = None
        self.is_authenticated = False

    def login(self, user):
        self.user = user
        self.is_authenticated = True

    def logout(self):
        self.user = None
        self.is_authenticated = False


session = SessionState()

# NEW PASTED -- WILL CHANGE
past_conversations = get_past_conversations()
# available_models = get_model_names()

# NEW PASTED -- WILL CHANGE


# Define the Gradio interface components
def create_app():
    # Initialize the database
    init_db()

    # Login form
    with gr.Blocks() as login_form:
        # gr.Label(value="Welcome to Intelligent Tutoring System!!", elem_id="main_heading", show_label=False)
        gr.Image("logo.png", show_label=False)
        login_email = gr.Textbox(label="Email")
        login_password = gr.Textbox(label="Password", type="password")
        login_message = gr.Textbox(label="Message", interactive=False, visible=False)
        login_button = gr.Button("Login",elem_id="login_button")
        register_link = gr.Button("Need an account? Register",elem_id="reg_button")

        def login_click(email, password):
            success, result = login_user(email, password)
            if success:
                session.login(result)
                return f"Welcome back, {result['first_name']} {result['last_name']}!"
            else:
                return result

        # login_button.click(login_click, [login_email, login_password], [login_message])

    # Registration form
    with gr.Blocks() as register_form:
        gr.Image("logo.png", show_label=False)
        gr.Markdown("# Register New Account!!",elem_id="main_heading")
        # gr.Label(value="Register New Account!!", elem_id="main_heading", show_label=False)
        
        # gr.Markdown("## Register New Account")
        first_name = gr.Textbox(label="First Name")
        last_name = gr.Textbox(label="Last Name")
        email = gr.Textbox(label="Email")
        password = gr.Textbox(label="Password", type="password")
        confirm_password = gr.Textbox(label="Confirm Password", type="password")
        register_message = gr.Textbox(label="Message", interactive=False)
        register_button = gr.Button("Register",elem_id="reg_button2")
        login_link = gr.Button("Already have an account? Login",elem_id="registered_button")

        def register_click(first_name, last_name, email, password, confirm_password):
            if password != confirm_password:
                return "Passwords do not match"

            success, message = register_user(first_name, last_name, email, password)
            return message

        register_button.click(
            register_click,
            [first_name, last_name, email, password, confirm_password],
            [register_message],
        )

        # Metrics Form
    with gr.Blocks() as metrics_form:
        gr.Image("logo.png", show_label=False)
        gr.Markdown("## Usage Statistics", elem_id="main_heading")
        with gr.Row(elem_id="row_with_white_bg"):
            with gr.Column(scale=1):
                gr.Markdown("User Name:", elem_id="markdowns_metrics")
                gr.Markdown("User Email:", elem_id="markdowns_metrics")
                gr.Markdown("Total Chat Count:", elem_id="markdowns_metrics")
                gr.Markdown("Average Chat Length:", elem_id="markdowns_metrics")
                gr.Markdown("Minimum Chat Length:", elem_id="markdowns_metrics")
                gr.Markdown("Maximum Chat Length:", elem_id="markdowns_metrics")
            with gr.Column(scale=1):
                user_name = gr.Markdown("**User Name**: N/A", elem_id="markdowns_metrics2")
                user_email = gr.Markdown("**Email**: N/A", elem_id="markdowns_metrics2")
                conversation_count = gr.Markdown("**Total Conversations**: 0", elem_id="markdowns_metrics2")
                avg_chat_length = gr.Markdown("**Average Chat Length**: 0", elem_id="markdowns_metrics2")
                min_chat_length = gr.Markdown("**Minimum Chat Length**: 0", elem_id="markdowns_metrics2")
                max_chat_length = gr.Markdown("**Maximum Chat Length**: 0", elem_id="markdowns_metrics2")
            with gr.Column(scale=2):
                conversation_trend_plot = gr.Plot(scale=1)
        with gr.Row(elem_id="row_with_white_bg"):
            with gr.Column():
                user_pie_plot = gr.Plot(scale=1)
            with gr.Column():
                model_usage_plot = gr.Plot(scale=1)    
        with gr.Row(elem_id="row_with_white_bg"):
            refresh_button = gr.Button("Refresh Metrics", elem_id="refresh_button")
            proceed_button = gr.Button("Go to Chat", elem_id="proceed_button")

        def get_metrics_and_charts():
            if not session.is_authenticated:
                return "N/A", "N/A", "0", None, None

            # Text metrics
            name = f"{session.user['first_name']} {session.user['last_name']}"
            email = session.user['email']
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute("""SELECT COUNT(*), round(avg(LENGTH(history)),2) avg_chat_length, min(LENGTH(history)) min_chat_length, 
            max(LENGTH(history)) max_chat_length FROM user_chats WHERE user_id = %s""", (session.user['user_id'],))
            row = cursor.fetchone()
            count_chats = row[0]
            avg_chat_length = row[1]
            min_chat_length = row[2]
            max_chat_length = row[3]

            # Pie Chart for User Usage
            cursor.execute("""SELECT 
                                SUM(CASE WHEN user_id = %s THEN 1 ELSE 0 END) AS user_chat_count,
                                SUM(CASE WHEN user_id != %s THEN 1 ELSE 0 END) AS other_users_chat_count
                            FROM user_chats """, (session.user['user_id'], session.user['user_id']))
            
            row = cursor.fetchone()
            user_chat_count = row[0] or 0  # Handle NULL case
            other_users_chat_count = row[1] or 0  # Handle NULL case

            cursor.execute("SELECT CONCAT(first_name,' ', last_name) FROM User WHERE user_id = %s", (session.user['user_id'],))
            name = cursor.fetchone()[0]
            
            labels = [f"{name} (You)", "Other Users"]
            sizes = [user_chat_count, other_users_chat_count]
            colors = ['#ff9999', '#66b3ff']  # Red for user, blue for others
            explode = (0.1, 0)  # Slightly explode the user's slice
            plt.figure(figsize=(4, 4))
            plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
            plt.title(f"Chat Usage: {name} vs. Other Users")
            plt.axis('equal')  # Equal aspect ratio ensures pie is circular
            pie_plot = plt.gcf()

            # Bar chart: Conversations per model
            # Assuming user_chats has a 'model' column; adjust query as needed
            cursor.execute(
                "select replace(model_name,'Eshita-ds/','') model_name, g_eval_score from cot_models"
            )
            email_data = cursor.fetchall()
            if email_data:
                models, g_eval_score = zip(*email_data)
            else:
                models, g_eval_score = ["No Data"], [0]
            plt.figure(figsize=(6, 4))
            colors = ['skyblue', 'lightgreen', 'salmon', 'gold']
            plt.bar(models, g_eval_score, color=colors)
            plt.xlabel("Model")
            plt.ylabel("G-Eval Scores")
            plt.title("Model Evaluation Scores")
            plt.xticks(rotation=15, ha='right')
            plt.tight_layout()
            model_plot = plt.gcf()

            # Line chart: Conversations per day (last 30 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            cursor.execute(
                """
                SELECT DATE(create_time), COUNT(*)
                FROM user_chats
                WHERE user_id = %s AND create_time >= %s
                GROUP BY DATE(create_time)
                """,
                (session.user['user_id'], start_date)
            )
            trend_data = cursor.fetchall()
            dates = [start_date + timedelta(days=i) for i in range(30)]
            counts = [0] * 30
            if trend_data:
                for date, count in trend_data:
                    if date:
                        delta = (date - start_date.date()).days
                        if 0 <= delta < 30:
                            counts[delta] = count
            plt.figure(figsize=(8, 4))
            plt.plot([d.strftime("%Y-%m-%d") for d in dates], counts, marker='o', color='teal')
            plt.xlabel("Date")
            plt.ylabel("Number of Conversations")
            plt.title("Conversation Trend (Last 30 Days)")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            trend_plot = plt.gcf()

            conn.close()
            return name, email, str(count_chats), str(avg_chat_length), str(min_chat_length), str(max_chat_length), model_plot, trend_plot, pie_plot
            
    # Protected content (only visible to authenticated users)
    with gr.Blocks() as protected_content:
        # gr.Markdown("Welcome to Protected Content")

        """def get_user_info():
            return f"Welcome, {session.user['first_name']} {session.user['last_name']}! Your email is {session.user['email']}"
            # return "Not authenticated"""

        # user_info = gr.Label(value="Hello, Welcome to Intelligent Tutoring System", elem_id="main_heading", show_label=False)
        gr.Image("logo.png", show_label=False)
        with gr.Row(elem_id="row_with_white_bg"):
            with gr.Column(scale=1, elem_id="column_with_grey_bg"):
                gr.Markdown("### Past Conversations",elem_id="markdowns")
                # past_conversations = get_past_conversations()
                available_models_dict = get_model_names()
                available_models = list(available_models_dict.values())
                #print(available_models, flush=True)
                chat_list = gr.Dropdown(choices=[(f"Conversation {key}", key) for key in past_conversations.keys()], label="Select Conversation")
                # model_list = gr.Dropdown(choices=[(name, key) for key, name in available_models.items()], label="Select Model", allow_custom_value=True)
                model_list = gr.Dropdown(choices=available_models, value=available_models[0] if available_models else None, label="Select Model", allow_custom_value=True, interactive=True)
                # model_list = gr.Radio(available_models, label="Select Model")
                with gr.Column(elem_id="column_with_white_bg"):
                    load_button = gr.Button("Load Chat",elem_id="chat_button")
                    new_chat_button = gr.Button("Start New Chat",elem_id="chat_button")
                    logout_button = gr.Button("Logout",elem_id="logout_button")

                def logout_click():
                    session.logout()
                    return "You have been logged out"
                
            
            with gr.Column(scale=3):
                with gr.Column(elem_id="column_with_grey_bg"):
                    chatbot = gr.Chatbot()
                    user_input = gr.Textbox(label="Type your message")
                with gr.Column(elem_id="column_with_white_bg2"):
                    send_button = gr.Button("Send",elem_id="send_button")
                
        def load_selected_chat(session_label):
            # session_id = session_label.split(" ")[-1]
            #print(session_label)
            # session_id = session_label.split("(")[-1].split(")")[0].replace("Session ID: ", "").strip()
            session_id = session_label.split(" (ID: ")[-1].replace(")", "").strip()
            #print(f"session id: {session_id}", flush=True)
            history = load_chat(session_id)
            session.user["session_id"] = session_id
            return history if history else [], session_label
        
        def start_new_chat():
            new_session_id = generate_session_id()
            session.user["session_id"] = new_session_id
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(buffered=True)
            
            # cursor.execute("INSERT INTO chats (session_name, history) VALUES (%s, %s)", ("Conversation", json.dumps([])))
            cursor.execute("""INSERT INTO user_chats (user_id, session_id, history, create_time) 
                           VALUES (%s, %s, %s, %s)""", (session.user["user_id"], new_session_id, json.dumps([]), datetime.now()))
            conn.commit()
            conn.close()
            updated_conversations_dict = get_past_conversations()
            updated_conversations = list(updated_conversations_dict.values())
            #print(updated_conversations)
            return gr.Dropdown(choices=updated_conversations, value=updated_conversations[-1], allow_custom_value=True), []
        
        load_button.click(load_selected_chat, inputs=[chat_list], outputs=[chatbot, chat_list])
        send_button.click(chatbot_response, inputs=[user_input, chatbot, chat_list, model_list], outputs=[chatbot, chatbot, user_input])
        new_chat_button.click(start_new_chat, outputs=[chat_list, chatbot])
        
        logout_button.click(logout_click, [], [login_message])

    # Main application with navigation between screens
    with gr.Blocks(css=custom_css) as app:
        nav_state = gr.State("login")

        with gr.Group(visible=True) as login_group:
            login_form.render()

        with gr.Group(visible=False) as register_group:
            register_form.render()

        with gr.Group(visible=False) as metrics_group:
            metrics_form.render()
            
        with gr.Group(visible=False) as protected_group:
            protected_content.render()

        def navigate_to_register(current_state):
            return {
                login_group: gr.update(visible=False),
                register_group: gr.update(visible=True),
                metrics_group: gr.update(visible=False),
                protected_group: gr.update(visible=False),
                nav_state: "register",
            }

        def navigate_to_login(current_state):
            return {
                login_group: gr.update(visible=True),
                register_group: gr.update(visible=False),
                metrics_group: gr.update(visible=False),
                protected_group: gr.update(visible=False),
                nav_state: "login",
            }

        def navigate_to_metrics(current_state):
            if session.is_authenticated:
                name, email, count, avg_length, min_length, max_length, model_plot, trend_plot, pie_plot = get_metrics_and_charts()
                return {
                    login_group: gr.update(visible=False),
                    register_group: gr.update(visible=False),
                    metrics_group: gr.update(visible=True),
                    protected_group: gr.update(visible=False),
                    nav_state: "metrics",
                    user_name: gr.update(value=name),
                    user_email: gr.update(value=email),
                    conversation_count: gr.update(value=count),
                    avg_chat_length: gr.update(value=avg_length),
                    min_chat_length: gr.update(value=min_length),
                    max_chat_length: gr.update(value=max_length),
                    model_usage_plot: gr.update(value=model_plot),
                    conversation_trend_plot: gr.update(value=trend_plot),
                    user_pie_plot: gr.update(value=pie_plot),
                    # chat_list: []
                }
            return {
                login_group: gr.update(visible=True),
                register_group: gr.update(visible=False),
                metrics_group: gr.update(visible=False),
                protected_group: gr.update(visible=False),
                nav_state: "login",
                user_name: gr.update(value="**User Name**: N/A"),
                user_email: gr.update(value="**Email**: N/A"),
                conversation_count: gr.update(value="**Total Conversations**: 0"),
                avg_chat_length: gr.update(value="**Average Chat Length**: 0"),
                min_chat_length: gr.update(value="**Minimum Chat Length**: 0"),
                max_chat_length: gr.update(value="**Maximum Chat Length**: 0"),
                model_usage_plot: gr.update(value=None),
                conversation_trend_plot: gr.update(value=None),
                user_pie_plot: gr.update(value=None),
            }
        
        def navigate_to_protected(current_state):
            if session.is_authenticated:
                past_conversations_dict = get_past_conversations()
                past_conversations = list(past_conversations_dict.values())

                #print(past_conversations, flush=True)
                return {
                    login_group: gr.update(visible=False),
                    register_group: gr.update(visible=False),
                    metrics_group: gr.update(visible=False),
                    protected_group: gr.update(visible=True),
                    nav_state: "protected",
                    chat_list: gr.update(choices=past_conversations, value=past_conversations[0], allow_custom_value=True),
                    # user_info: get_user_info(), ## Added new
                }
            else:
                return {
                    login_group: gr.update(visible=True),
                    register_group: gr.update(visible=False),
                    metrics_group: gr.update(visible=False),
                    protected_group: gr.update(visible=False),
                    nav_state: "login",
                }

        register_link.click(
            navigate_to_register,
            [nav_state],
            [login_group, register_group, protected_group, nav_state],
        )
        login_button.click(login_click, [login_email, login_password], [login_message])
        
        login_link.click(
            navigate_to_login,
            [nav_state],
            [login_group, register_group, protected_group, nav_state],
        )

        proceed_button.click(
            navigate_to_protected,
            [nav_state],
            [login_group, register_group, metrics_group, protected_group, nav_state, chat_list],
        )

        refresh_button.click(
            navigate_to_metrics,
            [nav_state],
            [login_group, register_group, metrics_group, protected_group, nav_state, user_name, 
             user_email, conversation_count, avg_chat_length, min_chat_length, max_chat_length, 
             model_usage_plot, conversation_trend_plot, user_pie_plot],
        )
        
        # Auto-navigate to protected content after successful login
        def auto_navigate_after_login(message):
            if "Welcome back" in message:
                # return navigate_to_protected(None)
                return navigate_to_metrics(None)
            else:
                return {
                    login_group: gr.update(visible=True),
                    register_group: gr.update(visible=False),
                    metrics_group: gr.update(visible=False),
                    protected_group: gr.update(visible=False),
                    nav_state: "login",
                    user_name: gr.update(value="**User Name**: N/A"),
                    user_email: gr.update(value="**Email**: N/A"),
                    conversation_count: gr.update(value="**Total Conversations**: 0"),
                    avg_chat_length: gr.update(value="**Average Chat Length**: 0"),
                    min_chat_length: gr.update(value="**Minimum Chat Length**: 0"),
                    max_chat_length: gr.update(value="**Maximum Chat Length**: 0"),
                    model_usage_plot: gr.update(value=None),
                    conversation_trend_plot: gr.update(value=None),
                    user_pie_plot: gr.update(value=None),
                }
        
        '''def auto_navigate_after_logout(message):
            # After logout, navigate to login screen
            return {
                login_group: gr.update(visible=True),
                register_group: gr.update(visible=False),
                protected_group: gr.update(visible=False),
                nav_state: "login"
            }'''
        
        login_message.change(
            auto_navigate_after_login,
            [login_message],
            [login_group, register_group, metrics_group, protected_group, nav_state, user_name, 
             user_email, conversation_count, avg_chat_length, min_chat_length, max_chat_length, model_usage_plot, 
             conversation_trend_plot, user_pie_plot] # chat_list, ] # user_info]
        )

        # Auto-navigate to login after logout
        logout_button.click(
            navigate_to_login,
            [nav_state],
            [login_group, register_group, metrics_group, protected_group, nav_state],
        )

    return app


# Launch the app
if __name__ == "__main__":
    app = create_app()
    app.launch()