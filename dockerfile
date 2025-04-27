FROM python:3.10-slim

# working directory
WORKDIR /app

COPY requirements.txt ./

# Install dependencies
RUN pip install --user --no-cache-dir --upgrade pip && \
    pip install --user  --no-cache-dir -r requirements.txt

COPY ./code ./CodeReviewer

WORKDIR /app/CodeReviewer/code/sh


CMD ["sleep", "infinity"]