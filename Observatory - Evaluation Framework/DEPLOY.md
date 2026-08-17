# Deploying Observatory to AWS

ALB → one Fargate task → RDS Postgres, with EFS holding uploads and the Chroma
index. Infrastructure is AWS CDK (Python) in [`infra/`](infra/).

```
        Internet
            │
      ┌─────▼─────┐
      │    ALB    │  public subnets, idle timeout 300s
      └─────┬─────┘
            │  /healthz
   ┌────────▼─────────┐
   │  Fargate task    │  2 vCPU / 8 GB, private subnet, 1 task
   │  uvicorn :8000   │
   └───┬──────────┬───┘
       │          │
 ┌─────▼────┐  ┌──▼──────────────┐
 │ RDS      │  │ EFS /data       │
 │ Postgres │  │  ├─ uploads/    │
 │ (16.4)   │  │  └─ index/      │
 └──────────┘  └─────────────────┘
```

## Prerequisites

- AWS credentials with permission to create VPC, ECS, RDS, EFS, ALB, Secrets Manager, IAM
- Docker running locally (CDK builds and pushes the image during `cdk deploy`)
- Node.js (for the CDK CLI) and Python 3.12

## First deploy

```bash
cd "Observatory - Evaluation Framework/infra"

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
npm install -g aws-cdk

export CDK_DEFAULT_REGION=ap-south-1        # or your region
cdk bootstrap                                # once per account+region
cdk deploy
```

`cdk deploy` prints an IAM change summary and waits for approval. It takes
15–25 minutes on a cold account — RDS provisioning dominates, and the image
build installs the full ML stack.

## Second step: the app is deployed but not yet working

The stack creates the secret **empty on purpose** — real keys in CDK source
would be committed to this public repository. The first task will start and fail
its `/readyz` check until you fill it in:

```bash
aws secretsmanager put-secret-value \
  --secret-id observatory/app \
  --secret-string '{
    "GROQ_API_KEY":"gsk_...",
    "GROQ_API_KEY_2":"gsk_...",
    "LANGCHAIN_API_KEY":"",
    "COLLINEAR_API_KEY":""
  }'

# Restart the task so it picks the new values up — secrets are read at start.
aws ecs update-service --cluster <cluster> --service <service> --force-new-deployment
```

Then open the `ServiceUrl` from the stack outputs. `GET /readyz` returns 200 with
`{"database": true, "groq_keys": 2}` once it is genuinely ready; a 503 there
tells you which dependency is still missing.

## Do not raise `desired_count` without reading this

The stack pins the service to **one task**, and that is a correctness
constraint, not a cost saving:

- **ChromaDB is SQLite on disk.** Several tasks writing one SQLite file across
  NFS is a corruption risk — NFS locking is not what SQLite assumes. A second
  task is a data hazard, not extra throughput.
- **BM25 and the retrieval bundle are cached in-process** (`_INDEX_CACHE` in
  `backend/services/rag_evaluator.py`). A second task shares none of it and
  re-chunks every document on its first request.

To scale horizontally, move Chroma to a server deployment (or swap the vector
store for OpenSearch/pgvector) and make the BM25 index shared. Until then one
task is correct — this is an internal tool serving a handful of analysts.

## Cost

Rough monthly, ap-south-1, running continuously:

| Component | Est. |
|---|---|
| Fargate 2 vCPU / 8 GB | ~$60 |
| RDS db.t4g.small + 20 GB | ~$30 |
| ALB | ~$20 |
| NAT gateway | ~$35 |
| EFS (low tens of GB) | ~$5 |
| **Total** | **~$150/mo** |

The NAT gateway is pure overhead for a service that only needs egress to Groq
and HuggingFace. If cost matters more than isolation, putting the task in a
public subnet with `assign_public_ip=True` removes it — at the price of the task
being directly addressable.

Stopping the service between evaluation campaigns (`desired_count=0`) drops the
Fargate line to zero while keeping data on EFS and RDS.

## Operations

```bash
# Logs
aws logs tail /aws/ecs/observatory --follow

# Roll back — the circuit breaker auto-reverts a deployment that never
# becomes healthy, but this forces it
aws ecs update-service --cluster <cluster> --service <service> \
  --task-definition <previous-revision>

# Redeploy after a code change
cd infra && cdk deploy
```

## Teardown

```bash
cdk destroy
```

RDS has `deletion_protection=True` and a `SNAPSHOT` removal policy, and EFS is
`RETAIN` — both survive `cdk destroy` deliberately, so an accidental teardown
cannot destroy evaluation history. Delete them by hand once you are certain.

## Known first-boot behaviour

- The app runs `create_all()` and `safe_migrate()` at **import time**, so the
  container exits if Postgres is unreachable. ECS restarts it; the 180s health
  check grace period exists to cover this plus ONNX model loading.
- Default embedding and reranker models are baked into the image. Any other
  model a user picks in the UI downloads on first use (~70–500 MB) onto the
  task's ephemeral disk and is lost on restart.
