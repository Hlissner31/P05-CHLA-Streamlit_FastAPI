# backend/Dockerfile

FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run the app using uvicorn
CMD ["uvicorn", "Backend:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
