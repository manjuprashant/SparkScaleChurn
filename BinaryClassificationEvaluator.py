from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import os
import sys

# -------------------------
# Initialize Spark session
# -------------------------
spark = SparkSession.builder \
    .appName("BinaryClassification") \
    .getOrCreate()

# -------------------------
# Define data paths
# -------------------------
train_path = "/opt/project/train_data.parquet"
test_path = "/opt/project/test_data.parquet"

# -------------------------
# Check if files exist
# -------------------------
if not os.path.exists(train_path):
    print(f"ERROR: Train file not found at {train_path}")
    sys.exit(1)

if not os.path.exists(test_path):
    print(f"ERROR: Test file not found at {test_path}")
    sys.exit(1)

# -------------------------
# Load the data
# -------------------------
train_data = spark.read.parquet(train_path)
test_data = spark.read.parquet(test_path)

# -------------------------
# Define the classifier
# -------------------------
classifier = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=10
)

# -------------------------
# Train the model
# -------------------------
model = classifier.fit(train_data)

# -------------------------
# Make predictions
# -------------------------
predictions = model.transform(test_data)

# -------------------------
# Evaluate
# -------------------------
evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

roc_auc = evaluator.evaluate(predictions)
print(f"ROC AUC Score: {roc_auc:.6f}")

# -------------------------
# Stop Spark session
# -------------------------
spark.stop()