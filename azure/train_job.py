"""Run src/train.py on an Azure ML compute cluster and register the resulting model."""
import sys
import time
from pathlib import Path

from azure.ai.ml import Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import AmlCompute, Environment, Model
from azure.core.exceptions import ResourceNotFoundError

from common import COMPUTE_NAME, MODEL_NAME, get_client

ROOT = Path(__file__).resolve().parent.parent


def ensure_compute(ml_client):
    try:
        ml_client.compute.get(COMPUTE_NAME)
    except ResourceNotFoundError:
        print(f"Creating compute cluster {COMPUTE_NAME} ...")
        cluster = AmlCompute(
            name=COMPUTE_NAME,
            size="Standard_DS3_v2",
            min_instances=0,  # scales to zero when idle, so you only pay while jobs run
            max_instances=2,
            idle_time_before_scale_down=300,
        )
        ml_client.compute.begin_create_or_update(cluster).result()


def main():
    ml_client = get_client()
    ensure_compute(ml_client)

    env = Environment(
        name="sklearn-train-env",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
        conda_file=str(ROOT / "environment" / "conda.yaml"),
    )

    job = command(
        code=str(ROOT / "src"),
        command="python train.py --n_estimators ${{inputs.n_estimators}} --model_output ${{outputs.model_output}}",
        inputs={"n_estimators": 200},
        outputs={"model_output": Output(type=AssetTypes.MLFLOW_MODEL)},
        environment=env,
        compute=COMPUTE_NAME,
        experiment_name="breast-cancer-classifier",
        display_name="train-random-forest",
    )

    job = ml_client.jobs.create_or_update(job)
    print(f"Submitted job {job.name}\nStudio (logs live here): {job.studio_url}")
    register(ml_client, wait_for_job(ml_client, job.name))


def wait_for_job(ml_client, job_name):
    # Poll instead of jobs.stream(): streaming needs storage account keys,
    # which new workspaces disable by default.
    status = None
    while status not in ("completed", "failed", "canceled"):
        # Depending on SDK version, status is "Completed" or JobStatus.COMPLETED.
        new_status = str(ml_client.jobs.get(job_name).status).split(".")[-1].lower()
        if new_status != status:
            print(f"  {time.strftime('%H:%M:%S')}  {new_status}")
            status = new_status
        if status not in ("completed", "failed", "canceled"):
            time.sleep(30)
    if status != "completed":
        raise SystemExit(f"Job {job_name} ended as {status}. Check the logs in Studio.")
    return job_name


def register(ml_client, job_name):
    model = ml_client.models.create_or_update(Model(
        name=MODEL_NAME,
        path=f"azureml://jobs/{job_name}/outputs/model_output",
        type=AssetTypes.MLFLOW_MODEL,
        description="Random forest trained by azure/train_job.py",
    ))
    print(f"Registered model {model.name} version {model.version}")


if __name__ == "__main__":
    # python azure/train_job.py               -> submit a new job, wait, register
    # python azure/train_job.py <job-name>    -> wait for an existing job, register
    if len(sys.argv) > 1:
        client = get_client()
        register(client, wait_for_job(client, sys.argv[1]))
    else:
        main()
