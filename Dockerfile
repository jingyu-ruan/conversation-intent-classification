### Instruction : Run in the project root directory
###  "docker build -t symtrain-assistant .""



# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system-level dependencies if needed (optional)
# RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy dependency list and install Python packages
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Make sure Python can find the src/ package
ENV PYTHONPATH=/app/src

# Expose Streamlit default port
EXPOSE 8501

# Set environment variable so Streamlit does not try to open a browser
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Command to run the Streamlit app
# If app.py is inside src/symtrain_assistant/
CMD ["streamlit", "run", "src/symtrain_assistant/app.py", "--server.port=8501", "--server.address=0.0.0.0"]