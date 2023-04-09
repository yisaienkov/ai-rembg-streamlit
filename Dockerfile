FROM python:3.10.11-slim

RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6 -y && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN python -m pip install -U pip && \
    python -m pip install -r requirements.txt && \
    python -m pip cache purge

COPY ./ /app/

WORKDIR /app/

CMD streamlit run src/main.py