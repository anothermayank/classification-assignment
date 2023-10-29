# Use Python runtime as a parent image
FROM python:3.8

# Set working directory in the container
WORKDIR ./app

# Install needed packages 
RUN pip install --no-cache-dir -r requirements.txt

# Expose a port
EXPOSE 80

# Define command to run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
