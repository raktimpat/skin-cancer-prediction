FROM python:3.11-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ./models /app/models
COPY ./network.py /app/network.py
COPY ./main.py /app/main.py



CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]