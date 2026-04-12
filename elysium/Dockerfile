FROM python:3.12-slim

# System tools available inside the container
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    wget \
    git \
    jq \
    vim \
    nano \
    unzip \
    zip \
    ffmpeg \
    imagemagick \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .

# Workspace volume — files saved here persist
RUN mkdir -p /workspace

ENV PYTHONUNBUFFERED=1

CMD ["python", "bot.py"]
