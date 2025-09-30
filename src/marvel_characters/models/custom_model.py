from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModelContext
from mlflow.utils.environment import _mlflow_conda_env

from marvel_characters.config import Tags

# We define this function above with the MarvelModelWrapper to package it together with the model
def adjust_predictions(predictions: np.ndarray | list[int]) -> dict[str, list[str]]:
    """Adjust predictions to human-readable format."""
    return {"Survival prediction": ["alive" if pred == 1 else "dead" for pred in predictions]}

# Inherits from mlflow.pyfunc.PythonModel
class MarvelModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper for LightGBM model."""

    def load_context(self, context: PythonModelContext) -> None:
        """Load the LightGBM model."""

        # mlflow.pyfunc.PythonModel has this method which actually loads the model.
        # It passes in context which contains all artifacts.
        # This line tells the wrapper to:
        # Find the underlying LightGBM model in the artifacts dict (logged earlier as "lightgbm-pipeline").
        # Reload it using mlflow.sklearn.load_model().
        # ✅ Now your wrapper has access to the underlying basic LightGBM pipeline.
        self.model = mlflow.sklearn.load_model(context.artifacts["lightgbm-pipeline"])

    def predict(self, context: PythonModelContext, model_input: pd.DataFrame | np.ndarray) -> dict:
        """Predict the survival of a character."""
        predictions = self.model.predict(model_input)
        return adjust_predictions(predictions)

    # Purpose: Log this wrapper itself as a new MLflow model in the pyfunc flavor.
    # Details:
    # python_model=self → log THIS wrapper (MarvelModelWrapper) instead of a plain sklearn model.
    # artifacts → attaches the underlying LightGBM pipeline model (wrapped_model_uri).
    # signature → logs input/output schema (input_example → prediction dict).
    # code_paths + conda_env → bundle the python package (marvel_characters.whl) so the wrapper code can be run anywhere, even when deployed on a different cluster or environment.
    def log_register_model(
        self,
        wrapped_model_uri: str,
        pyfunc_model_name: str,
        experiment_name: str,
        tags: Tags,
        code_paths: list[str],
        input_example: pd.DataFrame,
    ) -> None:
        """Log and register the model.

        :param wrapped_model_uri: URI of the wrapped model
        :param pyfunc_model_name: Name of the PyFunc model
        :param experiment_name: Name of the experiment
        :param tags: Tags for the model
        :param code_paths: List of code paths
        :param input_example: Input example for the model
        """
        # 1️⃣ Installation of the custom package wheel file
        # code_paths = paths to your Python package wheel files (marvel_characters.whl) containing custom code (MarvelModelWrapper).
        # This ensures the environment MLflow logs says:
        # "When deploying this model, don't forget to install THIS wheel file." ✅

        # conda_env = environment spec including those extra pip dependencies.

        # 2️⃣ Why do we need this?
        # Because your MarvelModelWrapper is custom code — not a builtin library like scikit-learn or pandas.

        # If you only log the model without attaching your custom package, then:

        # On your laptop: it works, because you already have marvel_characters in your Python env.
        # On Databricks Serving, a new cluster, or MLflow REST API: it would fail ⛔ because the serving environment doesn’t know about MarvelModelWrapper.
        # ✅ By bundling the .whl file into dependencies, MLflow ensures that wherever the model is deployed, it can pip-install your wrapper package so loading works.

        mlflow.set_experiment(experiment_name=experiment_name)
        with mlflow.start_run(run_name=f"wrapper-lightgbm-{datetime.now().strftime('%Y-%m-%d')}", tags=tags.to_dict()):
            additional_pip_deps = []
            for package in code_paths:
                whl_name = package.split("/")[-1]
                additional_pip_deps.append(f"code/{whl_name}")
            conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)

            # * artifact name lightgbm-pipeline
            # When saving this PyFunc wrapper, also include another model as an artifact.
            # Store it under the alias lightgbm-pipeline.”

            # That artifact is actually your underlying sklearn LightGBM pipeline model (the one you trained/logged earlier with mlflow.sklearn.log_model).
            signature = infer_signature(model_input=input_example, model_output={"Survival prediction": ["alive"]})
            model_info = mlflow.pyfunc.log_model(
                python_model=self,
                name="pyfunc-wrapper",
                artifacts={"lightgbm-pipeline": wrapped_model_uri}, # lightgbm-pipeline is the name of the folder where the model is stored, we can say this
                signature=signature,
                code_paths=code_paths,
                conda_env=conda_env,
            )

        # Pushes the logged pyfunc model into the Unity Catalog Model Registry.
        client = MlflowClient()
        registered_model = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=pyfunc_model_name,
            tags=tags.to_dict(),
        )
        latest_version = registered_model.version
        client.set_registered_model_alias(
            name=pyfunc_model_name,
            alias="latest-model",
            version=latest_version,
        )
        return latest_version
