# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variable for non-interactive installation (optional but useful)
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container at /app
COPY . .

# Expose port 8000 so that the container can be accessed on that port
EXPOSE 8000

# Define environment variables if needed (for example, AIPROXY_TOKEN)
# ENV AIPROXY_TOKEN=your_token_here

# Run the application using uvicorn when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
