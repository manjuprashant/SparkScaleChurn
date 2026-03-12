# -------------------------------------------------------------
# ADVANCED SPARK ML PIPELINE
# Includes:
# Confusion Matrix
# ROC Curve
# Precision Recall Curve
# Feature Importance
# DecisionTree vs RandomForest
# Hyperparameter Tuning
# Metrics Dashboard
# PDF Report Generation
# -------------------------------------------------------------

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet


# -------------------------------------------------------------
# 1 Create Spark Session
# -------------------------------------------------------------

spark = SparkSession.builder \
    .appName("Advanced Spark ML Pipeline") \
    .getOrCreate()


# -------------------------------------------------------------
# 2 Load Dataset
# -------------------------------------------------------------

print("Loading dataset...")

data = spark.read.csv(
    "data/telecom_data.csv",
    header=True,
    inferSchema=True
)


# -------------------------------------------------------------
# 3 Feature Engineering
# -------------------------------------------------------------

feature_cols = [
    "call_duration",
    "data_usage",
    "complaints",
    "billing_amount"
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

data = assembler.transform(data)


# -------------------------------------------------------------
# 4 Train Test Split
# -------------------------------------------------------------

train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

print("Training records:", train_data.count())
print("Testing records:", test_data.count())


# -------------------------------------------------------------
# 5 Train Decision Tree
# -------------------------------------------------------------

print("Training Decision Tree...")

dt = DecisionTreeClassifier(
    labelCol="churn",
    featuresCol="features"
)

dt_model = dt.fit(train_data)

dt_predictions = dt_model.transform(test_data)


# -------------------------------------------------------------
# 6 Train Random Forest
# -------------------------------------------------------------

print("Training Random Forest...")

rf = RandomForestClassifier(
    labelCol="churn",
    featuresCol="features",
    numTrees=100
)

rf_model = rf.fit(train_data)

rf_predictions = rf_model.transform(test_data)


# -------------------------------------------------------------
# 7 Evaluation Metrics
# -------------------------------------------------------------

accuracy_eval = MulticlassClassificationEvaluator(
    labelCol="churn",
    predictionCol="prediction",
    metricName="accuracy"
)

f1_eval = MulticlassClassificationEvaluator(
    labelCol="churn",
    predictionCol="prediction",
    metricName="f1"
)

auc_eval = BinaryClassificationEvaluator(
    labelCol="churn",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

dt_accuracy = accuracy_eval.evaluate(dt_predictions)
rf_accuracy = accuracy_eval.evaluate(rf_predictions)

dt_f1 = f1_eval.evaluate(dt_predictions)
rf_f1 = f1_eval.evaluate(rf_predictions)

dt_auc = auc_eval.evaluate(dt_predictions)
rf_auc = auc_eval.evaluate(rf_predictions)


# -------------------------------------------------------------
# 8 Metrics Dashboard
# -------------------------------------------------------------

metrics = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest"],
    "Accuracy": [dt_accuracy, rf_accuracy],
    "F1 Score": [dt_f1, rf_f1],
    "AUC": [dt_auc, rf_auc]
})

print(metrics)

metrics.to_csv("metrics_dashboard.csv", index=False)


# -------------------------------------------------------------
# 9 Confusion Matrix
# -------------------------------------------------------------

print("Generating Confusion Matrix...")

pdf = rf_predictions.select("churn", "prediction").toPandas()

cm = confusion_matrix(pdf["churn"], pdf["prediction"])

plt.figure()

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.colorbar()

plt.savefig("confusion_matrix.png")


# -------------------------------------------------------------
# 10 ROC Curve
# -------------------------------------------------------------

print("Generating ROC Curve...")

prob_df = rf_predictions.select("probability", "churn").toPandas()

scores = prob_df["probability"].apply(lambda x: float(x[1]))

fpr, tpr, _ = roc_curve(prob_df["churn"], scores)

roc_auc = auc(fpr, tpr)

plt.figure()

plt.plot(fpr, tpr, label="AUC = %0.2f" % roc_auc)

plt.plot([0,1], [0,1], "--")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig("roc_curve.png")


# -------------------------------------------------------------
# 11 Precision Recall Curve
# -------------------------------------------------------------

print("Generating Precision Recall Curve...")

precision, recall, _ = precision_recall_curve(prob_df["churn"], scores)

plt.figure()

plt.plot(recall, precision)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision Recall Curve")

plt.savefig("precision_recall_curve.png")


# -------------------------------------------------------------
# 12 Feature Importance
# -------------------------------------------------------------

print("Generating Feature Importance...")

importance = rf_model.featureImportances.toArray()

plt.figure()

plt.bar(feature_cols, importance)

plt.title("Feature Importance")

plt.xticks(rotation=45)

plt.savefig("feature_importance.png")


# -------------------------------------------------------------
# 13 Hyperparameter Optimization
# -------------------------------------------------------------

print("Running Hyperparameter Tuning...")

pipeline = Pipeline(stages=[assembler, rf])

paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50,100,200]) \
    .addGrid(rf.maxDepth, [5,10,15]) \
    .build()

crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=BinaryClassificationEvaluator(labelCol="churn"),
    numFolds=3
)

cv_model = crossval.fit(train_data)

best_model = cv_model.bestModel


# -------------------------------------------------------------
# 14 Save Optimized Model
# -------------------------------------------------------------

print("Saving optimized model...")

best_model.write().overwrite().save("optimized_random_forest_model")


# -------------------------------------------------------------
# 15 Generate PDF Report
# -------------------------------------------------------------

print("Generating PDF report...")

styles = getSampleStyleSheet()

story = []

story.append(Paragraph("Advanced Spark ML Evaluation Report", styles['Title']))

story.append(Spacer(1,20))

story.append(Paragraph("Decision Tree Accuracy: " + str(dt_accuracy), styles['Normal']))

story.append(Paragraph("Random Forest Accuracy: " + str(rf_accuracy), styles['Normal']))

story.append(Spacer(1,20))

story.append(Image("confusion_matrix.png",400,300))

story.append(Spacer(1,20))

story.append(Image("roc_curve.png",400,300))

story.append(Spacer(1,20))

story.append(Image("precision_recall_curve.png",400,300))

story.append(Spacer(1,20))

story.append(Image("feature_importance.png",400,300))

pdf = SimpleDocTemplate("advanced_spark_ml_report.pdf")

pdf.build(story)


# -------------------------------------------------------------
# 16 Stop Spark
# -------------------------------------------------------------

print("Pipeline Completed Successfully")

spark.stop()