FROM python:3.10-slim AS builder

ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
WORKDIR $APP_HOME

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . $APP_HOME

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
