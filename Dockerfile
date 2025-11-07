FROM python:3.10-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1

COPY . /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip wheel setuptools \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 7860
CMD ["python", "app.py"]
