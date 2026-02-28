FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace/ScientificResearchAssistant

COPY ScientificResearchAssistant/requirements.txt /tmp/requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt \
    && python -m nltk.downloader punkt stopwords

COPY ScientificResearchAssistant /workspace/ScientificResearchAssistant

WORKDIR /workspace/ScientificResearchAssistant/data/raw/scripts

CMD ["bash"]
