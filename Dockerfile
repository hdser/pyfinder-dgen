# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the working directory contents into the container at /app
COPY . .

# Ensure the output directory exists
RUN mkdir -p /app/output

# Set environment variables (optional)
ENV OUTPUT_DIR=/app/output

# Run data_generator.py when the container launches
CMD ["python", "data_generator.py", "--config_file", "config.json", "--output_base_dir", "/app/output"]
