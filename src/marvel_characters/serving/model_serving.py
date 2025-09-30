"""Model serving module for Marvel characters."""

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)


class ModelServing:
    """Manages model serving in Databricks for Marvel characters."""

    def __init__(self, model_name: str, endpoint_name: str) -> None:
        """Initialize the Model Serving Manager.

        :param model_name: Name of the model to be served
        :param endpoint_name: Name of the serving endpoint
        """
        self.workspace = WorkspaceClient()
        self.endpoint_name = endpoint_name
        self.model_name = model_name

    def get_latest_model_version(self) -> str:
        """Retrieve the latest version of the model.

        :return: Latest version of the model as a string
        """
        # It uses the mlflow.MlflowClient to connect to the MLflow Model Registry.
        # Crucially, it uses get_model_version_by_alias to look for the model version 
        # that has the "latest-model" alias assigned to it. This is a best practice, 
        # as it ensures you are always deploying the version that has been explicitly marked as the latest, not just the one with the highest version number.
        client = mlflow.MlflowClient()
        latest_version = client.get_model_version_by_alias(self.model_name, alias="latest-model").version
        return latest_version

    def deploy_or_update_serving_endpoint(
        self, version: str = "latest", workload_size: str = "Small", scale_to_zero: bool = True
    ) -> None:
        """Deploy or update the model serving endpoint in Databricks for Marvel characters.

        :param version: Model version to serve (default: "latest")
        :param workload_size: Size of the serving workload (default: "Small")
        :param scale_to_zero: Whether to enable scale-to-zero (default: True)
        """

        # If you pass "latest" (default): it fetches the UC model alias "latest-model" → resolves to the newest version
        endpoint_exists = any(item.name == self.endpoint_name for item in self.workspace.serving_endpoints.list())
        entity_version = self.get_latest_model_version() if version == "latest" else version

        # * Which UC model to serve (entity_name = model_name).
        # * Which version (entity_version = resolved #).
        # * Workload size = how much compute (Small, Medium, Large).
        # * scale_to_zero_enabled → whether endpoint auto-scales to 0 workers when idle (cost-saving).
        served_entities = [
            ServedEntityInput(
                entity_name=self.model_name,
                scale_to_zero_enabled=scale_to_zero,
                workload_size=workload_size,
                entity_version=entity_version,
            )
        ]

        # create() → sets up a new serving endpoint when one doesn’t yet exist.
        # update_config() → modifies an existing endpoint (e.g., swap in a new model version, adjust resources)
        if not endpoint_exists:
            self.workspace.serving_endpoints.create(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(
                    served_entities=served_entities,
                ),
            )
        else:
            self.workspace.serving_endpoints.update_config(name=self.endpoint_name, served_entities=served_entities)
