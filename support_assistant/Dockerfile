# Use Python 3.11 with a small Linux base image.
FROM python:3.11-slim

# Set /app as the working folder inside the container.
WORKDIR /app

# Install system tools needed by some Python packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependency file into the container.
COPY requirements.txt .

# Install all required Python packages.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container.
COPY . .

# Use mock mode by default for the graded offline pipeline.
ENV MOCK_LLM=1

# Make the FastAPI port available from the container.
EXPOSE 7860

# Start the FastAPI application when the container runs.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]