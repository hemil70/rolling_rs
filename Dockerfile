# Use a lightweight Python base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker caching for dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code (app.py and templates folder)
COPY . .

# Expose the port the app will run on (default Flask port)
EXPOSE 5000

# Set environment variables for Flask (production mode, no debug)
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Run the Flask app with Gunicorn for production-grade serving
# (Binds to all interfaces on port 5000, with 4 worker processes for handling requests)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
