FROM python:3.11-slim-bookworm

WORKDIR /donkeycar

COPY . .

RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libopenblas-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install -e .[pi]

CMD ["python3", "manage.py", "drive"]