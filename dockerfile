# Use an official Python runtime as a parent image
# Using 3.10 based on the warning message you shared earlier
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
# Assumes requirements.txt is in the same directory as the Dockerfile
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size and --upgrade pip first
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
# Assumes your code is primarily within the 'CodeReviewer' directory
COPY ./CodeReviewer ./CodeReviewer

# You might want to set the working directory to where your main script is
WORKDIR /app/CodeReviewer/code/sh 

RUN chmod +x finetune-ref.sh

# Add any other setup steps if needed

# (No CMD or ENTRYPOINT is specified as you only asked for setup)
# You would add a CMD or ENTRYPOINT later to run your application, e.g.:
# CMD ["python", "./CodeReviewer/code/test_model.py"]
CMD ["./finetune-ref.sh"]
