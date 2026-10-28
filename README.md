# ML on Azure ML: train → register → deploy

A scikit-learn classifier (breast-cancer dataset by default) that is trained on Azure ML compute,
registered as an MLflow model, and served from a managed online endpoint.

```
src/train.py          training script (runs locally or in Azure ML)
environment/conda.yaml training environment for the cloud job
azure/common.py       workspace connection + names
azure/train_job.py    creates compute, submits the training job, registers the model
azure/deploy.py       creates endpoint + deployment, sends a test request
azure/predict.py      sends rows from a CSV to the live endpoint
azure/cleanup.py      deletes the endpoint (stops billing)
data/new_patients.csv sample input for predict.py
```

## 1. One-time Azure setup

1. Install the [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli), then sign in:
   ```bash
   az login
   ```
2. Create a resource group and an Azure ML workspace (or create them in the Azure portal):
   ```bash
   az extension add -n ml
   az group create -n rg-ml-demo -l eastus
   az ml workspace create -n mlw-demo -g rg-ml-demo
   ```
3. Tell the scripts which workspace to use, either by downloading `config.json` from the workspace
   overview page in the portal into the project root, or by setting environment variables:
   ```powershell
   $env:AZURE_SUBSCRIPTION_ID = "<subscription-id>"
   $env:AZURE_RESOURCE_GROUP  = "rg-ml-demo"
   $env:AZUREML_WORKSPACE     = "mlw-demo"
   ```

## 2. Local setup and a local training run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
```

Metrics print to the console and the model is saved to `outputs/model`. Run `mlflow ui` to browse runs.

## 3. Train in Azure ML

```bash
python azure/train_job.py
```

This creates a `cpu-cluster` that scales to zero when idle, runs the job (the first run takes a few
minutes to build the environment image), streams logs, and registers the model as `breast-cancer-rf`.
Metrics appear in Azure ML Studio under the `breast-cancer-classifier` experiment.

## 4. Deploy

```bash
python azure/deploy.py
```

Endpoint names must be unique within an Azure region. If creation fails on the name, set
`$env:AZUREML_ENDPOINT_NAME = "breast-cancer-<your-initials>"` and re-run.
Deployment takes 10–20 minutes. The script prints a prediction and the scoring URI.

Then send your own rows:

```bash
python azure/predict.py data/new_patients.csv
```

### Troubleshooting deployment

| Error | Fix |
|---|---|
| `SubscriptionNotRegistered` | `az provider register -n` for `Microsoft.PolicyInsights`, `Microsoft.Network`, `Microsoft.Compute`, `Microsoft.Cdn` |
| `Not enough quota` | VM cores × instances × 1.2 must fit your quota. Default VM is `Standard_DS2_v2`; override with `AZUREML_INSTANCE_TYPE`, or request more quota in the portal |
| `Liveness probe failed: 502` | Read the container logs: `az ml online-deployment get-logs -n blue -e <endpoint> -g <rg> -w <ws>` |
| `has not been created successfully` | Delete the failed endpoint/deployment and deploy again |

Call it from anywhere:

```bash
curl -X POST <scoring-uri> -H "Authorization: Bearer <key>" -H "Content-Type: application/json" -d @azure/sample_request.json
```

## 5. Clean up (important)

A managed online endpoint bills by the hour while it exists. Delete it when you are done:

```bash
python azure/cleanup.py
```

## Using your own data

Put a CSV with a label column somewhere and pass it to the trainer:

```bash
python src/train.py --data path/to/data.csv --target label_column
```

For cloud runs, upload the CSV as a data asset and add it to the job's `inputs` in `azure/train_job.py`
(`Input(type="uri_file", path="azureml:<asset-name>@latest")`, then `--data ${{inputs.data}}` in the command).
Update `write_sample_request` in `azure/deploy.py` to build a request from your columns.
