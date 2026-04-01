from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder.appName("VectorAssemblerExample").getOrCreate()

# Load your processed feature dataset
df = spark.read.parquet("data/processed_features.parquet")

# Create feature vector
assembler = VectorAssembler(
    inputCols=[
        "avg_data_usage",
        "total_complaints",
        "avg_call_duration"
    ],
    outputCol="features"
)

# Transform dataset
final_df = assembler.transform(df)

# Check result
final_df.select("features", "churn").show(5)