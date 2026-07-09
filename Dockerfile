# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code (including your trained .pth weights)
COPY . .

# Expose the port your local inference server runs on
EXPOSE 8000

# Command to run the Flask web app
CMD ["python", "xray_flask/app.py"]