# Use Python 3.8 slim (matches your current base)
FROM python:3.8-slim-buster

# Install pip and dependencies in one layer for efficiency
WORKDIR /app

# Copy only requirements first (better Docker layer caching)
COPY requirements.txt .

# Install all packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application code
COPY . .

# Expose the port (optional but good practice)
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]