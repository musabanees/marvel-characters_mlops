# %%

import numpy as np
import pandas as pd

from dotenv import load_dotenv
import os

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow import MlflowClient

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score

load_dotenv()
databricks_host = os.getenv("DATABRICKS_HOST")
databricks_token = os.getenv("DATABRICKS_TOKEN")

workspace = WorkspaceClient(host=databricks_host, token=databricks_token)

# %%
# Define the catalog and schema name

catalog_name = "iris_basic_model"
schema_name = "iris_model_schema" #

# %% 
# --- Log the Metric and Model ---

mlflow.set_tracking_uri("databricks")
os.environ['DATABRICKS_HOST'] = databricks_host
os.environ['DATABRICKS_TOKEN'] = databricks_token

# 5. Set the experiment location
mlflow.set_experiment(f"/Users/<email>/Iris-UC-Experiment")

# Define the three-level name for our model in Unity Catalog
uc_model_name = f"{catalog_name}.{schema_name}.iris_logistic_regression"

# %%
# --- Load and split the Iris dataset ---

# Load and split the Iris dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# 1. Initialize the MLflow Client
client = MlflowClient()
# %%
# --- Model Development and logging with the SKlearn flavor ---

# Start an MLflow run
with mlflow.start_run(run_name="Iris Logistic Regression (UC)") as run:
    # Train the model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    # Make predictions to calculate accuracy AND infer the signature
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # <-- 2. INFER THE SIGNATURE FROM YOUR DATA
    # It looks at the training inputs (X_train) and the model's output (y_pred)
    signature = infer_signature(X_train, y_pred)
    
    # Log metrics and the model
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=uc_model_name, # this will log and register the model at the same time
        signature=signature  
    )
    
    print(f"\nModel '{uc_model_name}' logged to Unity Catalog with a signature.")
    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {accuracy:.2f}")

# COMMAND ----------
# --- Set the alias for the model ---

# The search result is ordered by version number descending, so the first result is the latest.
versions = client.search_model_versions(f"name='{uc_model_name}'")
latest_version = versions[0].version

print(f"Found latest version: {latest_version}")

# 3. Set the 'champion' alias for the new version
client.set_registered_model_alias(
    name=uc_model_name, 
    alias="champion", 
    version=latest_version
)

print(f"Alias 'champion' set for version {latest_version} of model '{uc_model_name}'.")



# COMMAND ----------
# --- This function will be packaged with our wrapper ---

def map_predictions_to_names(predictions: np.ndarray) -> dict[str, list[str]]:
    """Converts numeric predictions to string labels for the Iris dataset."""
    label_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    
    # The output is a dictionary for a clear, named output column
    return {"species": [label_map[pred] for pred in predictions]}
# COMMAND ----------
# --- Model Development and logging with the Pyfunc ---

class IrisModelWrapper(mlflow.pyfunc.PythonModel):
    """
    A custom pyfunc wrapper that loads a registered scikit-learn Iris model
    and adds a post-processing step to return string labels.
    """

    def load_context(self, context):
        """
        Loads the base scikit-learn model from the artifacts.
        """
        # The key "base_iris_model" must match the key we use when logging.
        self.model = mlflow.sklearn.load_model(context.artifacts["base_iris_model"])
        print("Base scikit-learn model loaded into the wrapper.")

    def predict(self, context, model_input: pd.DataFrame) -> dict:
        """
        Generates a prediction with the base model, then post-processes it.
        """
        numeric_predictions = self.model.predict(model_input)
        return map_predictions_to_names(numeric_predictions)

    def log_and_register(
        self,
        base_model_uri: str,
        wrapped_model_name: str,
        experiment_name: str,
        input_example: pd.DataFrame,
    ):
        """
        Logs this wrapper as a new model, registers it in Unity Catalog,
        and sets the 'champion' alias.
        """
        mlflow.set_experiment(experiment_name=experiment_name)
        with mlflow.start_run(run_name="Iris Pyfunc Wrapper") as run:

            # URI of the already-registered model we want to wrap.
            artifacts = {"base_iris_model": base_model_uri}
            
            # Infer the signature. The output now matches our post-processed dictionary.
            signature = infer_signature(
                model_input=input_example,
                model_output={"species": ["setosa"]} # Example output
            )

            # Log this wrapper class as a new pyfunc model
            model_info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=self,
                artifacts=artifacts,
                signature=signature,
            )

            # Register the newly logged wrapper model in Unity Catalog
            registered_model = mlflow.register_model(
                model_uri=model_info.model_uri,
                name=wrapped_model_name,
            )
            
            print(f"Successfully registered model '{wrapped_model_name}' version {registered_model.version}")

            # Set the 'champion' alias on the new version
            client = MlflowClient()
            client.set_registered_model_alias(
                name=wrapped_model_name,
                alias="champion",
                version=registered_model.version,
            )
            print(f"Alias 'champion' set for version {registered_model.version}.")
# COMMAND ----------
# --- Define the URI of the BASE model we want to wrap. ---

client = MlflowClient()

# 1. Define the URI of the BASE model we want to wrap.
# This points to the latest version of our previously registered sklearn model.
base_model_name = f"{catalog_name}.{schema_name}.iris_logistic_regression"
latest_version = client.search_model_versions(f"name='{base_model_name}'")[0].version
print(f"Found latest version of base model: {latest_version}")

# 3. Create a URI with the specific version number
base_model_uri = f"models:/{base_model_name}/{latest_version}"
print(f"Using specific model URI: {base_model_uri}")

# 2. Define the name for the NEW wrapped model we are about to create.
wrapped_model_name = f"{catalog_name}.{schema_name}.iris_human_readable_model"

# 3. Define the experiment where this run will be logged.
experiment_name = f"/Users/masab.a@turing.com/Iris-UC-Experiment"

# 4. Create an input example for the signature
# This is just for defining the schema, not for training.
input_example = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)

# --- Execution ---

# 5. Instantiate our wrapper
iris_wrapper = IrisModelWrapper()

# 6. Call the method to log and register the wrapper
iris_wrapper.log_and_register(
    base_model_uri=base_model_uri,
    wrapped_model_name=wrapped_model_name,
    experiment_name=experiment_name,
    input_example=input_example.head(5), # Just need a few rows for the schema
)

print("\nProcess complete. A new wrapped model has been registered.")


# COMMAND ----------
# ---- Serving the endpoint ----

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

class ModelServing:
    """
    A helper class to programmatically manage Databricks Model Serving endpoints.
    """
    def __init__(self, model_name: str, endpoint_name: str):
        self.workspace = WorkspaceClient() # Automatically authenticates
        self.model_name = model_name
        self.endpoint_name = endpoint_name

    def deploy_or_update_endpoint(
        self,
        model_alias: str = "champion", 
        workload_size: str = "Small", 
        scale_to_zero: bool = True
    ):
        """Deploys or updates the model serving endpoint."""
        
        # Check if an endpoint with the same name already exists
        endpoint_exists = any(e.name == self.endpoint_name for e in self.workspace.serving_endpoints.list())
        
        # First, resolve the alias to a specific version number.
        model_version_details = client.get_model_version_by_alias(
            self.model_name, alias=model_alias
        )
        version_to_serve = model_version_details.version

        print(f"Resolved alias '{model_alias}' to version {version_to_serve} for model '{self.model_name}'")
        
        # Define the model to be served. Instead of resolving a version number,
        # we can directly tell the endpoint to serve the model with a specific alias.
        served_model = ServedEntityInput(
            entity_name=self.model_name,
            entity_version=version_to_serve,
            workload_size=workload_size,
            scale_to_zero_enabled=scale_to_zero,
        )

        if not endpoint_exists:
            print(f"Creating new endpoint '{self.endpoint_name}'...")
            self.workspace.serving_endpoints.create_and_wait(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(served_entities=[served_model]),
            )
            print("Endpoint created successfully.")
        else:
            print(f"Updating existing endpoint '{self.endpoint_name}'...")
            self.workspace.serving_endpoints.update_config_and_wait(
                name=self.endpoint_name,
                served_entities=[served_model],
            )
            print("Endpoint updated successfully.")

# COMMAND ----------

# Define the full name of our registered pyfunc model in Unity Catalog
model_name = f"{catalog_name}.{schema_name}.iris_human_readable_model"

# Define a unique name for our serving endpoint
endpoint_name = "iris-model-endpoint"

# --- Deployment ---

# 1. Instantiate our helper class
model_serving = ModelServing(model_name=model_name, endpoint_name=endpoint_name)

# 2. Deploy the model version with the 'champion' alias
# The class handles whether to create a new endpoint or update an existing one.
model_serving.deploy_or_update_endpoint(model_alias="champion")

print(f"\nDeployment process complete for endpoint '{endpoint_name}'.")
# COMMAND ----------
