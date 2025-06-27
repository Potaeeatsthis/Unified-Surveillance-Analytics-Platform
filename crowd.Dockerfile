FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY crowd.requirements.txt .

RUN pip install --no-cache-dir -r crowd.requirements.txt

COPY crowd.py .
COPY yolo11n.engine .

ENV PYTHONUNBUFFERED=1

CMD ["python", "crowd.py"]
