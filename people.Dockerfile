FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
COPY people.requirements.txt .

RUN pip install --no-cache-dir -r people.requirements.txt

COPY people.py .
COPY yolo11n.cache yolo11n.engine .

ENV PYTHONUNBUFFERED=1

CMD ["python", "people.py"]
