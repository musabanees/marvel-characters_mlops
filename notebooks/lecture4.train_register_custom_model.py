
# Databricks notebook source
# MAGIC %pip install ../dist/marvel_characters-0.1.0-py3-none-any.whl


# Databricks notebook source

import mlflow
from pyspark.sql import SparkSession

from marvel_characters.config import ProjectConfig, Tags
from marvel_characters.models.custom_model import MarvelModelWrapper
from importlib.metadata import version
from dotenv import load_dotenv
from mlflow import MlflowClient
import os

# Set up Databricks or local MLflow tracking
def is_databricks():
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

# COMMAND ----------
# If you have DEFAULT profile and are logged in with DEFAULT profile,
# skip these lines

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


#  Databricks Asset Bundle, is a Bundle is your "project-in-a-box," defined by the databricks.yml file. When you deploy your project using the bundle (from your command line or a CI/CD system), the bundle is aware of the Git repository it's in.
# It automatically captures the commit hash and branch name.
# The magic happens inside your databricks.yml file, where you can use special placeholders to pass this information to your training job.

config = ProjectConfig.from_yaml(config_path="../project_config_marvel.yml", env="dev")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "main"})
marvel_characters_v = version("marvel_characters")

code_paths=[f"../dist/marvel_characters-{marvel_characters_v}-py3-none-any.whl"]

# COMMAND ----------

# This is the low-level client API for interacting with the MLflow tracking server and Model Registry.
# With this object, you can:
    # * Search experiments
    # * Register models
    # * Fetch model versions
    # * Move models between stages (Staging → Production)
    # * Manage aliases (like latest-model)
client = MlflowClient()
wrapped_model_version = client.get_model_version_by_alias(
    name=f"{config.catalog_name}.{config.schema_name}.marvel_character_model_basic",
    alias="latest-model")
# Initialize model with the config path

# COMMAND ----------
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()
X_test = test_set[config.num_features + config.cat_features]

# COMMAND ----------
pyfunc_model_name = f"{config.catalog_name}.{config.schema_name}.marvel_character_model_custom"
wrapper = MarvelModelWrapper()

# Logs a new MLflow model using PyFunc flavor, where:
# * The sklearn LightGBM pipeline is logged as an artifact inside it (lightgbm-pipeline).
# * The wrapper adds custom .predict() logic (turning 0/1 into “alive/dead”).
# * Dependencies (code_paths, .whl) are bundled so it can be deployed anywhere.
# * The run is also logged into the specified experiment (experiment_name).
# This will create a new model version in the Model Registry under the specified name (pyfunc_model_name).
# wrapped_model_uri is the a pointer to the underlying LightGBM pipeline model.
wrapper.log_register_model(wrapped_model_uri=f"models:/{wrapped_model_version.model_id}",
                           pyfunc_model_name=pyfunc_model_name,
                           experiment_name=config.experiment_name_custom,
                           input_example=X_test[0:1],
                           tags=tags,
                           code_paths=code_paths)

# COMMAND ----------
# unwrap and predict
loaded_pufunc_model = mlflow.pyfunc.load_model(f"models:/{pyfunc_model_name}@latest-model")

unwraped_model = loaded_pufunc_model.unwrap_python_model()

# COMMAND ----------
unwraped_model.predict(context=None, model_input=X_test[0:1])
# COMMAND ----------
# another predict function with uri

loaded_pufunc_model.predict(X_test[0:1])
# COMMAND ----------
