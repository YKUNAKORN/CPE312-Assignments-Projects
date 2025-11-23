# Use a small Python base image
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Train model during image build
RUN python train_model.py

# Expose Flask port
EXPOSE 5001

# Use gunicorn for production
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5001", "app:app"]
