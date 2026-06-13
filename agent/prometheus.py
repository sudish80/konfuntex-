from prometheus_client import CollectorRegistry, generate_latest, Counter, Gauge, CONTENT_TYPE_LATEST
from flask import Response

registry = CollectorRegistry()
TRAINING_LOSS = Gauge('training_loss', 'Current training loss', registry=registry)
JOB_COUNT = Counter('jobs_total', 'Total jobs started', registry=registry)

def prometheus_metrics():
    return Response(generate_latest(registry), mimetype=CONTENT_TYPE_LATEST)
