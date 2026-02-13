# Use official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY src/ ./src/
COPY data/processed/scoring_model.pkl ./data/processed/

# Expose port 80 to the outside world
EXPOSE 80

# Environment variable for the model path (optional, ensuring app finds it)
# Our app.py uses relative path logic, so keeping the structure /app/src and /app/data works.

# Run app.py when the container launches
# We use uvicorn directly to run the FastAPI app
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "80"]
