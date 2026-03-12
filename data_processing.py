from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, sum

# Create Spark session connected to cluster
spark = SparkSession.builder \
    .appName("SparkScale Churn Data Processing") \
    .master("local[*]") \
    .getOrCreate()

print("Connected to Spark Cluster")

# Load telecom dataset
df = spark.read.csv("data/telecom_data.csv", header=True, inferSchema=True)
print("Schema Validation")
df.printSchema()

print("Total Rows:")
print(df.count())

print("Sample Data")
df.show(10)

# Basic Aggregations
print("Aggregated Metrics")

agg_df = df.groupBy().agg(
    avg("call_duration").alias("avg_call_duration"),
    avg("data_usage").alias("avg_data_usage"),
    sum("complaints").alias("total_complaints")
)

agg_df.show()
# Save processed dataset
df.write.mode("overwrite").csv("/opt/project/processed")

print("Processed data saved to /opt/project/processed")
df.write.mode("overwrite").option("header", True).csv("/opt/project/processed")

spark.stop()