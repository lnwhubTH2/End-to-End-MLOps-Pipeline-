FROM python:3.9-slim

WORKDIR /app

# Install all requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trained Challenger model
COPY model/ ./model/

# Copy API implementation
COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
