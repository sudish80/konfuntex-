import json, sys, os
sys.path.insert(0, '.')
from storage.database import init_db
from storage.jobs import JobStore
init_db()
js = JobStore()
jobs = js.list()
if jobs:
    j = jobs[-1]
    print(f'Job: {j.id}')
    print(f'Status: {j.status}')
    print(f'Model: {j.base_model}')
    print(f'Dataset: {j.dataset}')
    print(f'Steps: {j.steps_count}, Completed: {j.completed_steps}')
    print(f'Created: {j.created_at}')
    if j.logs:
        logs = json.loads(j.logs)
        for log in logs[-10:]:
            ts = log.get("timestamp", "")
            msg = log.get("message", "")
            print(f'  [{ts}] {msg}')
    if j.messages:
        msgs = json.loads(j.messages)
        for m in msgs[-10:]:
            role = m.get("role", "")
            content = str(m.get("content", ""))[:200]
            print(f'  [{role}] {content}')
else:
    print('No jobs found')
