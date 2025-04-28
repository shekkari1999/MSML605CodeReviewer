FROM python:3.10-slim

# working directory
WORKDIR /root/app

COPY requirements.txt ./

# Install dependencies
RUN pip install --user --no-cache-dir --upgrade pip && \
    pip install --user  --no-cache-dir -r requirements.txt

COPY ./code /root/app/CodeReviewer

ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /root/app/CodeReviewer/sh

# Start an interactive bash shell when the container runs
CMD ["sleep", "infinity"]
