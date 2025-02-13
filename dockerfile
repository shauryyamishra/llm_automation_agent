# Use a lightweight Python image
FROM python:3.9-slim

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
COPY . .

# Expose port 5000 (as the app runs on port 5000)
EXPOSE 5000

# Run the application using Python
CMD ["python", "app.py"]

# hi
