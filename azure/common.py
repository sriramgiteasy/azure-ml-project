"""Shared Azure ML settings and client."""
import os

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

COMPUTE_NAME = os.getenv("AZUREML_COMPUTE", "cpu-cluster")
MODEL_NAME = os.getenv("AZUREML_MODEL_NAME", "breast-cancer-rf")
ENDPOINT_NAME = os.getenv("AZUREML_ENDPOINT_NAME", "breast-cancer-ep")  # must be unique in the Azure region
# Azure ML reserves quota for 20% extra instances (rounded up), so 1 instance of a
# 2-core VM needs 4 cores of quota. DS2_v2 fits new subscriptions' 6-core limit.
INSTANCE_TYPE = os.getenv("AZUREML_INSTANCE_TYPE", "Standard_DS2_v2")


def get_client() -> MLClient:
    """Uses env vars if set, otherwise a config.json downloaded from the Azure ML workspace."""
    # `az` can take >10s (the SDK default) to return a token on Windows.
    credential = DefaultAzureCredential(process_timeout=60)
    sub = os.getenv("AZURE_SUBSCRIPTION_ID")
    if sub:
        return MLClient(
            credential,
            subscription_id=sub,
            resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
            workspace_name=os.environ["AZUREML_WORKSPACE"],
        )
    return MLClient.from_config(credential)
