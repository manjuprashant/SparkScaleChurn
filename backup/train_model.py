from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Create Spark session

spark = SparkSession.builder \
.appName("Churn Prediction Model") \
.getOrCreate()

print("Spark ML Pipeline Started")

# Load dataset

df = spark.read.csv("/opt/spark/data/telecom_data.csv", header=True, inferSchema=True)

print("Dataset Loaded")
df.printSchema()

# Feature columns

feature_columns = [
"call_duration",
"data_usage",
"complaints",
"billing_amount"
]

# Convert features into vector

assembler = VectorAssembler(
inputCols=feature_columns,
outputCol="features"
)

# Logistic Regression model

lr = LogisticRegression(
featuresCol="features",
labelCol="churn"
)

# Pipeline

pipeline = Pipeline(stages=[assembler, lr])

# Train/Test Split

train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

print("Training Data Count:", train_data.count())
print("Testing Data Count:", test_data.count())

# Train model

model = pipeline.fit(train_data)

print("Model Training Completed")

# Make predictions

predictions = model.transform(test_data)

print("Sample Predictions")
predictions.select("features", "churn", "prediction").show(10)

# -----------------------------

# CONFUSION MATRIX

# -----------------------------

print("Confusion Matrix")

tp = predictions.filter("prediction = 1 AND churn = 1").count()
tn = predictions.filter("prediction = 0 AND churn = 0").count()
fp = predictions.filter("prediction = 1 AND churn = 0").count()
fn = predictions.filter("prediction = 0 AND churn = 1").count()

print("True Positives:", tp)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)

# -----------------------------

# ACCURACY

# -----------------------------

accuracy_evaluator = MulticlassClassificationEvaluator(
labelCol="churn",
predictionCol="prediction",
metricName="accuracy"
)

accuracy = accuracy_evaluator.evaluate(predictions)

print("Model Accuracy:", accuracy)

# -----------------------------

# AUC EVALUATION

# -----------------------------

auc_evaluator = BinaryClassificationEvaluator(
labelCol="churn",
rawPredictionCol="rawPrediction",
metricName="areaUnderROC"
)

auc = auc_evaluator.evaluate(predictions)

print("Model AUC Score:", auc)

# -----------------------------

# SAVE MODEL

# -----------------------------

model.write().overwrite().save("/opt/spark/models/churn_model")

print("Model saved successfully")

spark.stop()
