from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import os
import sys

# ----------------------------
# Set PYTHONUSERBASE to avoid permission issues
# ----------------------------
user_base = "/opt/project/.local"
os.environ["PYTHONUSERBASE"] = user_base
os.environ["PYTHONPATH"] = f"{user_base}/lib/python3.8/site-packages:" + os.environ.get("PYTHONPATH", "")

# ----------------------------
# Initialize Spark
# ----------------------------
spark = SparkSession.builder.appName("ChurnBatchJob").getOrCreate()

# ----------------------------
# Load the persisted pipeline
# ----------------------------
pipeline_model_path = "/opt/project/spark_churn_pipeline"
pipeline_model = PipelineModel.load(pipeline_model_path)

# ----------------------------
# Load new monthly user data
# ----------------------------
new_data_path = "/opt/project/new_monthly_data.parquet"
if not os.path.exists(new_data_path):
    print(f"Error: New data not found at {new_data_path}", file=sys.stderr)
    sys.exit(1)

new_data = spark.read.parquet(new_data_path)

# ----------------------------
# Generate predictions
# ----------------------------
predictions = pipeline_model.transform(new_data)

# ----------------------------
# Select relevant columns and save
# ----------------------------
output_path = "/opt/project/predictions/monthly_churn_predictions.parquet"
predictions.select("user_id", "probability", "prediction") \
           .write.mode("overwrite") \
           .parquet(output_path)

print(f"Predictions saved at {output_path}")