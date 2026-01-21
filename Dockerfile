# Base image with Python 3.10
FROM python:3.10-slim

# Updating and installing required system packages for madmom
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Copy the code
COPY . /app

# Installing Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install flask librosa matplotlib mutagen soundfile numpy cython madmom

# Opening the port for Flask
EXPOSE 5000

# Starting Flask
CMD ["python", "app.py"]
