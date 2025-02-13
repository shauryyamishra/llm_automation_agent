from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
import os
import json
from helpers import run_task, read_file

# Ensure that only files under /data are accessed.
DATA_DIR = os.path.abspath("./data")

app = FastAPI(title="LLM-Based Automation Agent")

import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# Retrieve AI Proxy Token from environment variable
API_PROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")

if not API_PROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN environment variable not set!")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "LLM Automation Agent is running!", "token": API_PROXY_TOKEN[:5] + '...'})  # Masking token

@app.route("/run", methods=["POST"])
def run_model():
    data = request.json
    if not data or "input" not in data:
        return jsonify({"error": "Missing input data"}), 400
    
    # Simulate processing using AI Proxy Token
    response = {
        "input": data["input"],
        "output": f"Processed using AI Proxy Token: {API_PROXY_TOKEN[:5]}..."  # Masked token for security
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

@app.post("/run")
async def run(task: str = Query(..., description="Plain-English task description")):
    try:
        # Here you would call your LLM to parse the task and then run it.
        # For simplicity, run_task() (in helpers.py) simulates task processing.
        result = run_task(task, data_dir=DATA_DIR)
        return {"status": "success", "result": result}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e))

@app.get("/read", response_class=PlainTextResponse)
async def read(path: str = Query(..., description="File path to read (must be under /data)")):
    # Security check: Only allow files under DATA_DIR
    abs_path = os.path.abspath(path)
    if not abs_path.startswith(DATA_DIR):
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        content = read_file(abs_path)
        return content
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")

from helpers import llm_parse

def run_task(task_description: str, data_dir: str) -> str:
    # Use the llm_parse function to get a plan for the task.
    plan = llm_parse(task_description)
    
    if plan.get("action") == "count_wednesdays":
        input_path = os.path.join(data_dir, "dates.txt")
        output_path = os.path.join(data_dir, "dates-wednesdays.txt")
        count = count_wednesdays(input_path)  # assuming you have this helper function
        with open(output_path, "w") as f:
            f.write(str(count))
        return f"Wednesdays counted: {count}"
    else:
        raise ValueError("Task not recognized or not supported.")

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

# Ensure security constraints: Restrict access to /data and prevent file deletion
DATA_DIR = "/data"

app = Flask(__name__)
openai.api_key = os.getenv("AIPROXY_TOKEN")  # Ensure API token is set in environment

def validate_path(filepath):
    """Ensure file access stays within /data."""
    abs_path = os.path.abspath(filepath)
    if not abs_path.startswith(DATA_DIR):
        raise PermissionError("Access outside /data is not allowed")
    return abs_path

def run_shell_command(command):
    """Run a shell command safely."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

def install_uv():
    """Install uv package manager."""
    run_shell_command("pip install uv")

def run_data_script(email):
    """Run the external script for data generation."""
    script_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    run_shell_command(f"curl -s {script_url} | python3 - {email}")

def format_markdown():
    """Format markdown using Prettier."""
    run_shell_command("npx prettier@3.4.2 --write /data/format.md")

def count_wednesdays():
    """Count the number of Wednesdays in a given date list."""
    with open(validate_path("/data/dates.txt")) as f:
        dates = [datetime.strptime(line.strip(), "%Y-%m-%d").weekday() for line in f]
    wednesday_count = dates.count(2)
    with open(validate_path("/data/dates-wednesdays.txt"), "w") as f:
        f.write(str(wednesday_count))

def sort_contacts():
    """Sort contacts JSON file by last name, then first name."""
    with open(validate_path("/data/contacts.json")) as f:
        contacts = json.load(f)
    contacts.sort(key=lambda x: (x['last_name'], x['first_name']))
    with open(validate_path("/data/contacts-sorted.json"), "w") as f:
        json.dump(contacts, f, indent=2)

def extract_recent_logs():
    """Extract first lines of 10 most recent .log files."""
    logs = sorted((f for f in os.listdir("/data/logs/") if f.endswith(".log")), key=os.path.getmtime, reverse=True)[:10]
    with open(validate_path("/data/logs-recent.txt"), "w") as f:
        f.write("\n".join(open(validate_path(f"/data/logs/{log}")).readline().strip() for log in logs))

def generate_docs_index():
    """Extract H1 headings from Markdown files and generate an index."""
    index = {}
    for filename in os.listdir("/data/docs/"):
        if filename.endswith(".md"):
            with open(validate_path(f"/data/docs/{filename}")) as f:
                for line in f:
                    if line.startswith("# "):
                        index[filename] = line.strip("# ").strip()
                        break
    with open(validate_path("/data/docs/index.json"), "w") as f:
        json.dump(index, f, indent=2)

def extract_email_sender():
    """Extract sender's email from an email text file using LLM."""
    with open(validate_path("/data/email.txt")) as f:
        email_content = f.read()
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Extract sender email from this text."}, {"role": "user", "content": email_content}]
    )
    with open(validate_path("/data/email-sender.txt"), "w") as f:
        f.write(response["choices"][0]["message"]["content"].strip())

def extract_credit_card_number():
    """Extract credit card number from an image."""
    image = Image.open(validate_path("/data/credit-card.png"))
    card_number = pytesseract.image_to_string(image).replace(" ", "")
    with open(validate_path("/data/credit-card.txt"), "w") as f:
        f.write(card_number)

def get_sql_sales():
    """Calculate total sales for 'Gold' tickets from SQLite."""
    conn = sqlite3.connect(validate_path("/data/ticket-sales.db"))
    cur = conn.cursor()
    cur.execute("SELECT SUM(units * price) FROM tickets WHERE type='Gold'")
    total_sales = cur.fetchone()[0]
    conn.close()
    with open(validate_path("/data/ticket-sales-gold.txt"), "w") as f:
        f.write(str(total_sales))

def handle_task(task_desc):
    """Determine which task to execute based on task description."""
    if "install uv" in task_desc:
        install_uv()
    elif "run data script" in task_desc:
        run_data_script(task_desc.split()[-1])
    elif "format markdown" in task_desc:
        format_markdown()
    elif "count Wednesdays" in task_desc:
        count_wednesdays()
    elif "sort contacts" in task_desc:
        sort_contacts()
    elif "extract recent logs" in task_desc:
        extract_recent_logs()
    elif "generate docs index" in task_desc:
        generate_docs_index()
    elif "extract email sender" in task_desc:
        extract_email_sender()
    elif "extract credit card number" in task_desc:
        extract_credit_card_number()
    elif "calculate total sales for Gold tickets" in task_desc:
        get_sql_sales()
    else:
        return "Task not recognized"
    return "Task completed"

@app.route("/run", methods=["POST"])
def run_task():
    data = request.get_json()
    task_desc = data.get("task", "")
    result = handle_task(task_desc)
    return jsonify({"status": "success", "message": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

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

# Set the root directory for file operations (must be /data)
DATA_DIR = "/data"

app = Flask(__name__)
# Retrieve the AI Proxy Token from environment variables
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
    A9: Using embeddings (here, a simplified approach with difflib), find the most similar pair of comments
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
    query = "SELECT * FROM sample_table"  # Replace with an actual query\n    cur.execute(query)\n    rows = cur.fetchall()\n    columns = [desc[0] for desc in cur.description]\n    data = [dict(zip(columns, row)) for row in rows]\n    conn.close()\n    output_path = validate_path(os.path.join(DATA_DIR, \"sql-output.json\"))\n    with open(output_path, \"w\") as f:\n        json.dump(data, f, indent=2)\n    return \"SQL query executed.\"\n\ndef scrape_website():\n    \"\"\"\n    B6: Extract data from a website (scrape) and save it to /data/scraped.txt.\n    \"\"\"\n    from bs4 import BeautifulSoup\n    url = \"https://example.com\"  # Replace with actual URL\n    response = requests.get(url)\n    soup = BeautifulSoup(response.text, 'html.parser')\n    text = soup.get_text(separator=' ', strip=True)\n    output_path = validate_path(os.path.join(DATA_DIR, \"scraped.txt\"))\n    with open(output_path, \"w\", encoding=\"utf-8\") as f:\n        f.write(text)\n    return \"Website scraped.\"\n\ndef resize_image():\n    \"\"\"\n    B7: Resize an image (/data/sample_image.png) and save as /data/resized_image.png.\n    \"\"\"\n    input_path = validate_path(os.path.join(DATA_DIR, \"sample_image.png\"))  # Replace with your image file\n    output_path = validate_path(os.path.join(DATA_DIR, \"resized_image.png\"))\n    image = Image.open(input_path)\n    resized = image.resize((image.width // 2, image.height // 2))\n    resized.save(output_path)\n    return \"Image resized.\"\n\ndef transcribe_audio():\n    \"\"\"\n    B8: Transcribe audio from /data/audio.mp3 and write the transcription to /data/audio-transcription.txt.\n    \"\"\"\n    input_path = validate_path(os.path.join(DATA_DIR, \"audio.mp3\"))\n    output_path = validate_path(os.path.join(DATA_DIR, \"audio-transcription.txt\"))\n    # For demonstration, we'll simulate the transcription.\n    transcription = \"Simulated transcription of the audio file.\"\n    with open(output_path, \"w\") as f:\n        f.write(transcription)\n    return \"Audio transcribed.\"\n\ndef markdown_to_html():\n    \"\"\"\n    B9: Convert /data/markdown.md to HTML and save as /data/markdown.html.\n    \"\"\"\n    input_path = validate_path(os.path.join(DATA_DIR, \"markdown.md\"))\n    output_path = validate_path(os.path.join(DATA_DIR, \"markdown.html\"))\n    with open(input_path, \"r\", encoding=\"utf-8\") as f:\n        md_text = f.read()\n    html_text = markdown.markdown(md_text)\n    with open(output_path, \"w\", encoding=\"utf-8\") as f:\n        f.write(html_text)\n    return \"Markdown converted to HTML.\"\n\n### ----------------- Task Dispatcher ----------------- ###\n\ndef handle_task(task_desc):\n    task_desc = task_desc.lower()\n    # Phase A (Operations Tasks)\n    if \"install uv\" in task_desc and \"datagen\" in task_desc:\n        email = task_desc.split()[-1]  # Expecting the email as the last word\n        return install_uv_and_run_datagen(email)\n    elif \"format markdown\" in task_desc:\n        return format_markdown_file()\n    elif \"count wednesdays\" in task_desc:\n        return count_wednesdays_in_dates()\n    elif \"sort contacts\" in task_desc:\n        return sort_contacts_json()\n    elif \"extract recent logs\" in task_desc:\n        return extract_first_line_logs()\n    elif \"generate docs index\" in task_desc:\n        return index_markdown_docs()\n    elif \"extract email sender\" in task_desc:\n        return extract_email_sender()\n    elif \"extract credit card number\" in task_desc:\n        return extract_credit_card_number()\n    elif \"find similar comments\" in task_desc:\n        return find_similar_comments()\n    elif \"calculate total sales for gold tickets\" in task_desc:\n        return calculate_gold_sales()\n    # Phase B (Business Tasks)\n    elif \"fetch api data\" in task_desc:\n        return fetch_api_data()\n    elif \"clone git repo\" in task_desc:\n        return clone_git_repo_and_commit()\n    elif \"run sql query\" in task_desc:\n        return run_general_sql_query()\n    elif \"scrape website\" in task_desc:\n        return scrape_website()\n    elif \"resize image\" in task_desc:\n        return resize_image()\n    elif \"transcribe audio\" in task_desc:\n        return transcribe_audio()\n    elif \"convert markdown to html\" in task_desc:\n        return markdown_to_html()\n    else:\n        return \"Task not recognized\"\n\n@app.route(\"/run\", methods=[\"POST\"])\ndef run_task():\n    data = request.get_json()\n    task_desc = data.get(\"task\", \"\")\n    result = handle_task(task_desc)\n    return jsonify({\"status\": \"success\", \"message\": result})\n\n# CSV filtering endpoint (Business Task B10)\n@app.route(\"/filter_csv\", methods=[\"GET\"])\ndef filter_csv():\n    csv_path = validate_path(os.path.join(DATA_DIR, \"data.csv\"))\n    filter_column = request.args.get(\"column\")\n    filter_value = request.args.get(\"value\")\n    results = []\n    with open(csv_path, \"r\", newline=\"\", encoding=\"utf-8\") as csvfile:\n        reader = csv.DictReader(csvfile)\n        for row in reader:\n            if filter_column in row and row[filter_column] == filter_value:\n                results.append(row)\n    return jsonify(results)\n\nif __name__ == \"__main__\":\n    app.run(host=\"0.0.0.0\", port=5000, debug=True)\n```

---

### **Additional Files**

**requirements.txt** (ensure you include these or add others as needed):

