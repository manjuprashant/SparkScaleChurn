from pyspark.sql import SparkSession
from pyspark.sql.types import *

spark = SparkSession.builder.appName("ChurnDataIngestion").getOrCreate()

schema = StructType([
    StructField("user_id", IntegerType(), True),
    StructField("data_usage", DoubleType(), True),
    StructField("call_duration", DoubleType(), True),
    StructField("complaints", IntegerType(), True),
    StructField("churn", IntegerType(), True)
])

df = spark.read.csv("data/churn.csv", header=True, schema=schema)

df.printSchema()
print("Row Count:", df.count())