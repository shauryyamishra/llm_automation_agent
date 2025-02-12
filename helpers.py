import os
import subprocess
import datetime
import json

def run_task(task_description: str, data_dir: str) -> str:
    """
    Parses the task_description (using an LLM in a real scenario)
    and executes one or more internal steps.
    
    For demonstration, we handle one sample task:
    - Counting Wednesdays in a file (/data/dates.txt)
    """
    # VERY basic parsing – in a real agent, you would use an LLM to map task to function calls.
    if "dates.txt" in task_description and "Wednesday" in task_description:
        input_path = os.path.join(data_dir, "dates.txt")
        output_path = os.path.join(data_dir, "dates-wednesdays.txt")
        count = count_wednesdays(input_path)
        with open(output_path, "w") as f:
            f.write(str(count))
        return f"Wednesdays counted: {count}"
    # (Add more conditionals for each task A1-A10 and B3-B10)
    else:
        raise ValueError("Task not recognized or not supported.")

def count_wednesdays(file_path: str) -> int:
    """
    Reads the file at file_path, which should contain one date per line.
    Returns the number of dates that are Wednesdays.
    """
    count = 0
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    # Assuming date format is YYYY-MM-DD; adjust format as needed.
                    try:
                        dt = datetime.datetime.strptime(line, "%Y-%m-%d")
                        if dt.weekday() == 2:  # Monday=0, Wednesday=2
                            count += 1
                    except ValueError:
                        # Skip lines that do not match the expected format.
                        continue
    except FileNotFoundError:
        raise ValueError("Input file not found: " + file_path)
    return count

def read_file(file_path: str) -> str:
    with open(file_path, "r") as f:
        return f.read()

# Additional helper functions for:
# - Formatting Markdown using prettier (e.g., using subprocess to call prettier if installed)
# - Sorting JSON arrays from a file
# - Extracting first lines of log files
# - Building an index of Markdown files in /data/docs/
# - Passing email text to an LLM (stubbed function) to extract sender address
# - Passing images to an OCR+LLM pipeline (stubbed function)
# - Finding similar comments using embeddings (this may require additional libraries)
# - Running SQL queries on a SQLite database (using the sqlite3 module)

def llm_parse(task: str) -> dict:
    """
    (Stub) Simulate calling an LLM to parse a task.
    In practice, this function would send the task string to an LLM (using your AIPROXY_TOKEN)
    and return a JSON plan.
    """
    # For now, return a dummy plan indicating the 'count_wednesdays' action.
    return {
        "action": "count_wednesdays",
        "input_file": "/data/dates.txt",
        "output_file": "/data/dates-wednesdays.txt"
    }
