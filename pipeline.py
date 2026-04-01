from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier

# Start Spark session
spark = SparkSession.builder.appName("ChurnPipeline").getOrCreate()

# Load processed dataset (after feature engineering)
df = spark.read.parquet("data/processed_features.parquet")

# -------------------------------
# Step 1: Feature Vector Creation
# -------------------------------
assembler = VectorAssembler(
    inputCols=[
        "avg_data_usage",
        "total_complaints",
        "avg_call_duration"
    ],
    outputCol="features"
)

# -------------------------------
# Step 2: Model
# -------------------------------
rf = RandomForestClassifier(
    labelCol="churn",
    featuresCol="features",
    numTrees=50
)

# -------------------------------
# Step 3: Pipeline
# -------------------------------
pipeline = Pipeline(stages=[assembler, rf])

# Train pipeline
pipeline_model = pipeline.fit(df)

# -------------------------------
# Step 4: Save Pipeline
# -------------------------------
pipeline_model.write().overwrite().save("model/churn_pipeline")

print("✅ Pipeline trained and saved successfully")