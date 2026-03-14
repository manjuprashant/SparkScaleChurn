from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

spark = SparkSession.builder.appName("BatchPrediction").getOrCreate()

# Load trained model
pipeline_model = PipelineModel.load("/opt/project/models/churn_model")

# Load new data
new_data = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("/opt/project/data/telecom_data.csv")

# Run prediction
predictions = pipeline_model.transform(new_data)

# Save predictions
predictions.select("user_id","prediction","probability") \
    .write.mode("overwrite") \
    .parquet("/opt/project/predictions")

print("Predictions saved successfully!")

spark.stop()