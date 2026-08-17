#!/usr/bin/env python3
"""CDK entry point. Deploy with `cdk deploy` from this directory."""
import os

import aws_cdk as cdk

from observatory_stack import ObservatoryStack

app = cdk.App()

ObservatoryStack(
    app, "ObservatoryEvaluationFramework",
    env=cdk.Environment(
        account=os.getenv("CDK_DEFAULT_ACCOUNT"),
        region=os.getenv("CDK_DEFAULT_REGION", "ap-south-1"),
    ),
    description="Observatory — RAG Evaluation Framework (Fargate + RDS + EFS)",
)

cdk.Tags.of(app).add("Project", "Observatory")
cdk.Tags.of(app).add("ManagedBy", "CDK")

app.synth()
