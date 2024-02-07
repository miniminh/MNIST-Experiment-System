FROM python:3.9-slim-buster

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /code
WORKDIR /code

COPY ./requirements.txt .


RUN pip install -r requirements.txt 

COPY . .

EXPOSE 7504

RUN mkdir -p search_params
RUN mkdir -p jobs
RUN mkdir -p models

CMD ["python", "app.py"]
