FROM python:3.11-slim

WORKDIR /donkeycar

COPY . .

RUN apt-get update && apt-get install -y \
libhdf5-dev \
libatlas3-base \
git \
build-essential \
python3-dev \
libcap-dev \
&& rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install -e .[pi]

CMD ["python3", "manage.py", "drive"]
