"""Deploy the latest registered model to a managed online endpoint and send a test request."""
import json
from pathlib import Path

from azure.ai.ml.entities import ManagedOnlineDeployment, ManagedOnlineEndpoint
from sklearn.datasets import load_breast_cancer

from common import ENDPOINT_NAME, INSTANCE_TYPE, MODEL_NAME, get_client

SAMPLE_PATH = Path(__file__).resolve().parent / "sample_request.json"


def write_sample_request():
    X = load_breast_cancer(as_frame=True).data.head(3)
    payload = {"input_data": {"columns": list(X.columns), "index": list(range(len(X))), "data": X.values.tolist()}}
    SAMPLE_PATH.write_text(json.dumps(payload, indent=2))


def main():
    ml_client = get_client()
    model = ml_client.models.get(MODEL_NAME, label="latest")
    print(f"Deploying {model.name}:{model.version}")

    endpoint = ManagedOnlineEndpoint(name=ENDPOINT_NAME, auth_mode="key")
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    # MLflow models need no scoring script or environment: Azure ML generates both.
    deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=ENDPOINT_NAME,
        model=model,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    endpoint.traffic = {"blue": 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    write_sample_request()
    response = ml_client.online_endpoints.invoke(
        endpoint_name=ENDPOINT_NAME, deployment_name="blue", request_file=str(SAMPLE_PATH)
    )
    print("Prediction:", response)

    ep = ml_client.online_endpoints.get(ENDPOINT_NAME)
    print(f"Scoring URI: {ep.scoring_uri}")
    print("Get the key with: az ml online-endpoint get-credentials -n", ENDPOINT_NAME)


if __name__ == "__main__":
    main()
