# Use NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code
COPY asr_api.py .

# Expose the API port
EXPOSE 8001

# Set default runtime to GPU (via NVIDIA runtime)
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Run the FastAPI application
CMD ["uvicorn", "asr_api:app", "--host", "0.0.0.0", "--port", "8001"]
