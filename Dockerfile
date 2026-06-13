# ===== Build stage =====
FROM python:3.12-slim AS build

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt && \
    rm -rf /root/.cache/pip

# ===== Runtime stage =====
FROM python:3.12-slim AS runtime

RUN groupadd -r agent && useradd -r -g agent agent

WORKDIR /app

COPY --from=build /install /usr/local
COPY --chown=agent:agent . .

RUN mkdir -p /data && chown agent:agent /data

USER agent

EXPOSE 8080

ENV PYTHONPATH=/app \
    COLAB_AGENT_DATA_DIR=/data \
    PYTHONUNBUFFERED=1

VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["python", "-m", "agent.service", "--fastapi"]
