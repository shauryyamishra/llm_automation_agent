from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
import os
import json
from helpers import run_task, read_file

# Ensure that only files under /data are accessed.
DATA_DIR = os.path.abspath("./data")

app = FastAPI(title="LLM-Based Automation Agent")

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
