from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
from matplotlib.backends.backend_pdf import PdfPages

# Start Spark
spark = SparkSession.builder.appName("Model Evaluation").getOrCreate()

print("Starting Model Evaluation")

# Load dataset
df = spark.read.csv("/opt/spark/data/telecom_data.csv", header=True, inferSchema=True)

# Load trained model
model = PipelineModel.load("/opt/spark/models/churn_model")

print("Model Loaded")

# Split dataset
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Predict
predictions = model.transform(test_data)

# Convert to pandas
pdf = predictions.select("churn", "prediction", "probability").toPandas()

pdf["prob"] = pdf["probability"].apply(lambda x: float(x[1]))

y_true = pdf["churn"]
y_pred = pdf["prediction"]
y_prob = pdf["prob"]

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

TP = cm[1][1]
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# ROC
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

# Precision Recall
precisions, recalls, _ = precision_recall_curve(y_true, y_prob)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC:", roc_auc)

# Save metrics
metrics_text = f"""
Confusion Matrix
TP: {TP}
TN: {TN}
FP: {FP}
FN: {FN}

Accuracy: {accuracy}
Precision: {precision}
Recall: {recall}
F1 Score: {f1}
AUC: {roc_auc}
"""

with open("evaluation_metrics.txt", "w") as f:
    f.write(metrics_text)

print("Metrics saved to file")

# Store metrics for plots
metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
metric_values = [accuracy, precision, recall, f1, roc_auc]

# Create PDF report
with PdfPages("evaluation_report.pdf") as pdf_report:

    # Confusion Matrix Plot
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.colorbar()
    pdf_report.savefig()
    plt.close()

    # ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, label="AUC=" + str(round(roc_auc,3)))
    plt.plot([0,1],[0,1])
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    pdf_report.savefig()
    plt.close()

    # Precision Recall Curve
    plt.figure()
    plt.plot(recalls, precisions)
    plt.title("Precision Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    pdf_report.savefig()
    plt.close()

    # Metrics Comparison Bar Chart
    plt.figure()
    plt.bar(metric_names, metric_values)
    plt.title("Model Performance Metrics Comparison")
    plt.ylabel("Score")
    pdf_report.savefig()
    plt.close()

    # Metrics Line Plot
    plt.figure()
    plt.plot(metric_names, metric_values, marker='o')
    plt.title("Metrics Trend")
    plt.ylabel("Score")
    pdf_report.savefig()
    plt.close()

print("Evaluation PDF generated")

spark.stop()