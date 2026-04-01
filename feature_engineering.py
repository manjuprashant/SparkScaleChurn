from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, sum, count

# Start Spark session
spark = SparkSession.builder.appName("FeatureEngineering").getOrCreate()

# Load cleaned data
df = spark.read.parquet("data/cleaned_data.parquet")

# -----------------------------------------
# Feature Engineering using Spark SQL
# -----------------------------------------

# Create temporary view
df.createOrReplaceTempView("user_data")

# User-level aggregations
features_df = spark.sql("""
    SELECT
        user_id,
        AVG(data_usage) AS avg_data_usage,
        SUM(complaints) AS total_complaints,
        AVG(call_duration) AS avg_call_duration,
        COUNT(*) AS activity_count,
        MAX(churn) AS churn
    FROM user_data
    GROUP BY user_id
""")

# Show engineered features
features_df.show(5)

# Save processed features
features_df.write.mode("overwrite").parquet("data/processed_features.parquet")

print("✅ Feature engineering completed and saved")