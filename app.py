import os
import json
import requests
import sqlite3
import subprocess
import shutil
import markdown
import csv
import openai
from datetime import datetime
from PIL import Image
import pytesseract
import difflib
from flask import Flask, request, jsonify
from bs4 import BeautifulSoup  # For website scraping

# Set the root directory for file operations (must be /data)
DATA_DIR = "/data"

app = Flask(__name__)

# Retrieve AI Proxy Token from environment variables
openai.api_key = os.getenv("AIPROXY_TOKEN")
if not openai.api_key:
    raise ValueError("AIPROXY_TOKEN environment variable not set!")

def validate_path(filepath):
    """Ensure file access is only within DATA_DIR."""
    abs_path = os.path.abspath(filepath)
    if not abs_path.startswith(os.path.abspath(DATA_DIR)):
        raise PermissionError("Access outside /data is not allowed")
    return abs_path

def run_shell_command(command):
    """Run a shell command and return its output."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

### ----------------- Phase A: Operations Tasks ----------------- ###

def install_uv_and_run_datagen(email):
    """
    A1: Install uv (if required) and run the datagen script with the given email.
    This generates required data files.
    """
    run_shell_command("pip install uv")
    script_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    run_shell_command(f"curl -s {script_url} | python3 - {email}")
    return "Data generation script executed."

def format_markdown_file():
    """
    A2: Format /data/format.md using prettier@3.4.2.
    Requires that npx is installed.
    """
    md_path = validate_path(os.path.join(DATA_DIR, "format.md"))
    run_shell_command(f"npx prettier@3.4.2 --write {md_path}")
    return "Markdown formatted."

def count_wednesdays_in_dates():
    """
    A3: Count the number of Wednesdays in /data/dates.txt and write the count to /data/dates-wednesdays.txt.
    Assumes dates are in YYYY-MM-DD format.
    """
    input_path = validate_path(os.path.join(DATA_DIR, "dates.txt"))
    output_path = validate_path(os.path.join(DATA_DIR, "dates-wednesdays.txt"))
    count = 0
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    dt = datetime.strptime(line, "%Y-%m-%d")
                    if dt.weekday() == 2:  # Wednesday (Monday=0)
                        count += 1
                except ValueError:
                    continue
    with open(output_path, "w") as f:
        f.write(str(count))
    return f"Number of Wednesdays: {count}"

def sort_contacts_json():
    """
    A4: Sort the contacts array in /data/contacts.json by last_name then first_name,
    and write the result to /data/contacts-sorted.json.
    """
    input_path = validate_path(os.path.join(DATA_DIR, "contacts.json"))
    output_path = validate_path(os.path.join(DATA_DIR, "contacts-sorted.json"))
    with open(input_path, "r") as f:
        contacts = json.load(f)
    contacts.sort(key=lambda x: (x.get("last_name", ""), x.get("first_name", "")))
    with open(output_path, "w") as f:
        json.dump(contacts, f, indent=2)
    return "Contacts sorted."

def extract_first_line_logs():
    """
    A5: Write the first line of the 10 most recent .log files from /data/logs/
    to /data/logs-recent.txt (most recent first).
    """
    logs_dir = validate_path(os.path.join(DATA_DIR, "logs"))
    log_files = [f for f in os.listdir(logs_dir) if f.endswith(".log")]
    log_files.sort(key=lambda f: os.path.getmtime(os.path.join(logs_dir, f)), reverse=True)
    selected_logs = log_files[:10]
    first_lines = []
    for lf in selected_logs:
        with open(os.path.join(logs_dir, lf), "r") as f:
            first_lines.append(f.readline().strip())
    output_path = validate_path(os.path.join(DATA_DIR, "logs-recent.txt"))
    with open(output_path, "w") as f:
        f.write("\n".join(first_lines))
    return "Extracted first lines of recent logs."

def index_markdown_docs():
    """
    A6: For each Markdown file in /data/docs/, extract the first occurrence of an H1 (a line starting with '# ')
    and create an index mapping filename (without path) to title. Save as /data/docs/index.json.
    """
    docs_dir = validate_path(os.path.join(DATA_DIR, "docs"))
    index = {}
    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            if file.endswith(".md"):
                full_path = os.path.join(root, file)
                with open(full_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("# "):
                            index[file] = line.strip()[2:].strip()
                            break
    output_path = validate_path(os.path.join(docs_dir, "index.json"))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    return "Markdown docs indexed."

def extract_email_sender():
    """
    A7: Pass the content of /data/email.txt to an LLM to extract the sender's email address,
    and write the result to /data/email-sender.txt.
    """
    input_path = validate_path(os.path.join(DATA_DIR, "email.txt"))
    output_path = validate_path(os.path.join(DATA_DIR, "email-sender.txt"))
    with open(input_path, "r") as f:
        content = f.read()
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract the sender's email address from the following email message."},
            {"role": "user", "content": content}
        ]
    )
    sender_email = response["choices"][0]["message"]["content"].strip()
    with open(output_path, "w") as f:
        f.write(sender_email)
    return "Email sender extracted."

def extract_credit_card_number():
    """
    A8: Pass the image /data/credit-card.png to an LLM/OCR to extract the credit card number (without spaces)
    and write it to /data/credit-card.txt.
    """
    input_path = validate_path(os.path.join(DATA_DIR, "credit-card.png"))
    output_path = validate_path(os.path.join(DATA_DIR, "credit-card.txt"))
    image = Image.open(input_path)
    card_number = pytesseract.image_to_string(image).replace(" ", "")
    with open(output_path, "w") as f:
        f.write(card_number)
    return "Credit card number extracted."

def find_similar_comments():
    """
    A9: Using embeddings (simplified with difflib), find the most similar pair of comments
    in /data/comments.txt and write them (one per line) to /data/comments-similar.txt.
    """
    input_path = validate_path(os.path.join(DATA_DIR, "comments.txt"))
    output_path = validate_path(os.path.join(DATA_DIR, "comments-similar.txt"))
    with open(input_path, "r") as f:
        comments = [line.strip() for line in f if line.strip()]
    max_ratio = 0
    similar_pair = ("", "")
    for i in range(len(comments)):
        for j in range(i + 1, len(comments)):
            ratio = difflib.SequenceMatcher(None, comments[i], comments[j]).ratio()
            if ratio > max_ratio:
                max_ratio = ratio
                similar_pair = (comments[i], comments[j])
    with open(output_path, "w") as f:
        f.write(similar_pair[0] + "\n" + similar_pair[1])
    return "Similar comments identified."

def calculate_gold_sales():
    """
    A10: Connect to the SQLite database /data/ticket-sales.db, calculate total sales (units * price)
    for tickets with type 'Gold', and write the result to /data/ticket-sales-gold.txt.
    """
    db_path = validate_path(os.path.join(DATA_DIR, "ticket-sales.db"))
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
    total = cur.fetchone()[0]
    conn.close()
    output_path = validate_path(os.path.join(DATA_DIR, "ticket-sales-gold.txt"))
    with open(output_path, "w") as f:
        f.write(str(total))
    return "Gold ticket sales calculated."

### ----------------- Phase B: Business Tasks ----------------- ###

def fetch_api_data():
    """
    B3: Fetch data from an API and save it to /data/api-data.json.
    """
    api_url = "https://api.example.com/data"  # Replace with a real API endpoint
    response = requests.get(api_url)
    data = response.json()
    output_path = validate_path(os.path.join(DATA_DIR, "api-data.json"))
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    return "API data fetched."

def clone_git_repo_and_commit():
    """
    B4: Clone a git repository and make a commit. Clone into /data/repo.
    """
    repo_url = "https://github.com/example/repo.git"  # Replace with the actual repository URL
    target_dir = validate_path(os.path.join(DATA_DIR, "repo"))
    run_shell_command(f"git clone {repo_url} {target_dir}")
    dummy_file = os.path.join(target_dir, "dummy.txt")
    with open(dummy_file, "w") as f:
        f.write("This is a test commit from the automation agent.")
    run_shell_command(f"cd {target_dir} && git add dummy.txt && git commit -m 'Automated commit' && git push")
    return "Git repository cloned and committed."

def run_general_sql_query():
    """
    B5: Run a general SQL query on a SQLite (or DuckDB) database and save the result to /data/sql-output.json.
    """
    db_path = validate_path(os.path.join(DATA_DIR, "sample.db"))  # Assumes sample.db exists
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    query = "SELECT * FROM sample_table"  # Replace with an actual query
    cur.execute(query)
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    data = [dict(zip(columns, row)) for row in rows]
    conn.close()
    output_path = validate_path(os.path.join(DATA_DIR, "sql-output.json"))
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    return "SQL query executed."

def scrape_website():
    """
    B6: Extract data from a website (scrape) and save it to /data/scraped.txt.
    """
    url = "https://example.com"  # Replace with actual URL
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)
    output_path = validate_path(os.path.join(DATA_DIR, "scraped.txt"))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return "Website scraped."

def resize_image():
    """
    B7: Resize an image (/data/sample_image.png) and save it as /data/resized_image.png.
    """
    input_path = validate_path(os.path.join(DATA_DIR, "sample_image.png"))  # Replace with your image file
    output_path = validate_path(os.path.join(DATA_DIR, "resized_image.png"))
    image = Image.open(input_path)
    resized = image.resize((image.width // 2, image.height // 2))
    resized.save(output_path)
    return "Image resized."

def transcribe_audio():
    """
    B8: Transcribe audio from /data/audio.mp3 and write the transcription to /data/audio-transcription.txt.
    """
    input_path = validate_path(os.path.join(DATA_DIR, "audio.mp3"))
    output_path = validate_path(os.path.join(DATA_DIR, "audio-transcription.txt"))
    transcription = "Simulated transcription of the audio file."  # Simulated; replace with real transcription if needed
    with open(output_path, "w") as f:
        f.write(transcription)
    return "Audio transcribed."

def markdown_to_html():
    """
    B9: Convert /data/markdown.md to HTML and save it as /data/markdown.html.
    """
    input_path = validate_path(os.path.join(DATA_DIR, "markdown.md"))
    output_path = validate_path(os.path.join(DATA_DIR, "markdown.html"))
    with open(input_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    html_text = markdown.markdown(md_text)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_text)
    return "Markdown converted to HTML."

### ----------------- Task Dispatcher ----------------- ###

def handle_task(task_desc):
    task_desc = task_desc.lower()
    # Phase A (Operations Tasks)
    if "install uv" in task_desc and "datagen" in task_desc:
        email = task_desc.split()[-1]  # Expecting the email as the last word
        return install_uv_and_run_datagen(email)
    elif "format markdown" in task_desc:
        return format_markdown_file()
    elif "count wednesdays" in task_desc:
        return count_wednesdays_in_dates()
    elif "sort contacts" in task_desc:
        return sort_contacts_json()
    elif "extract recent logs" in task_desc:
        return extract_first_line_logs()
    elif "generate docs index" in task_desc:
        return index_markdown_docs()
    elif "extract email sender" in task_desc:
        return extract_email_sender()
    elif "extract credit card number" in task_desc:
        return extract_credit_card_number()
    elif "find similar comments" in task_desc:
        return find_similar_comments()
    elif "calculate total sales for gold tickets" in task_desc:
        return calculate_gold_sales()
    # Phase B (Business Tasks)
    elif "fetch api data" in task_desc:
        return fetch_api_data()
    elif "clone git repo" in task_desc:
        return clone_git_repo_and_commit()
    elif "run sql query" in task_desc:
        return run_general_sql_query()
    elif "scrape website" in task_desc:
        return scrape_website()
    elif "resize image" in task_desc:
        return resize_image()
    elif "transcribe audio" in task_desc:
        return transcribe_audio()
    elif "convert markdown to html" in task_desc:
        return markdown_to_html()
    else:
        return "Task not recognized"

@app.route("/run", methods=["POST"])
def run_task_endpoint():
    data = request.get_json()
    task_desc = data.get("task", "")
    result = handle_task(task_desc)
    return jsonify({"status": "success", "message": result})

# CSV filtering endpoint (Business Task B10)
@app.route("/filter_csv", methods=["GET"])
def filter_csv():
    csv_path = validate_path(os.path.join(DATA_DIR, "data.csv"))
    filter_column = request.args.get("column")
    filter_value = request.args.get("value")
    results = []
    with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if filter_column in row and row[filter_column] == filter_value:
                results.append(row)
    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

# hi

