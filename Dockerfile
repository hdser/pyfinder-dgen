# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Build arguments for UID and GID
ARG USER_ID=1000
ARG GROUP_ID=1000

# Create a group and user with the specified UID and GID
RUN addgroup --gid $GROUP_ID appgroup && \
    adduser --disabled-password --gecos "" --uid $USER_ID --gid $GROUP_ID appuser

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the working directory contents into the container at /app
COPY . .

# Ensure the output directory exists
RUN mkdir -p /app/output && chown appuser:appgroup /app/output

# Switch to the non-root user
USER appuser

# Set environment variables (optional)
ENV OUTPUT_DIR=/app/output

# Run data_generator.py when the container launches
CMD ["python", "data_generator.py", "--config_file", "config.json", "--output_base_dir", "/app/output"]
