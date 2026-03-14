
# SparkScale Churn Prediction

## Overview

SparkScale Churn Prediction is a distributed machine learning pipeline built using **Apache Spark** and **Docker** to predict customer churn in a telecom dataset.
The system demonstrates scalable data processing, model training, and batch prediction using a Spark cluster.

This project simulates a **production-style big data pipeline** where churn prediction is trained and executed across multiple worker nodes.

---

## Technologies Used

* Apache Spark 3.5
* Python (PySpark)
* Docker & Docker Compose
* Machine Learning (Spark ML)
* Parquet Data Format

---

## System Architecture

```
Telecom Dataset
      │
      ▼
Data Processing (PySpark)
      │
      ▼
Feature Engineering
      │
      ▼
Model Training (Spark ML)
      │
      ▼
Saved Model
      │
      ▼
Batch Prediction
      │
      ▼
Prediction Output
```

Running on a distributed Spark cluster:

```
Docker Spark Cluster
│
├── Spark Master
├── Spark Worker 1
└── Spark Worker 2
```

---

## Project Structure

```
SparkScale_Churn
│
├── data/                      # Raw telecom dataset
├── processed/                 # Processed Spark dataset
├── models/                    # Saved trained model
├── predictions/               # Prediction results
│
├── data_processing.py         # Data preprocessing pipeline
├── train_model.py             # Spark ML model training
├── batch_predict.py           # Batch prediction pipeline
├── evaluate.py                # Model evaluation
│
├── docker-compose.yml         # Spark cluster configuration
├── train_data.parquet         # Training dataset
└── evaluation_report_churn.pdf
```

---

## Spark Cluster Setup

Start the Spark cluster using Docker:

```bash
docker compose up -d
```

Check running containers:

```bash
docker ps
```

Spark Master UI:

```
http://localhost:8080
```

---

## Train the Model

Enter the Spark master container:

```bash
docker exec -it spark-master bash
```

Run the training pipeline:

```bash
/opt/spark/bin/spark-submit \
--master spark://spark-master:7077 \
/opt/project/train_model.py
```

---

## Run Batch Prediction

```bash
/opt/spark/bin/spark-submit \
--master spark://spark-master:7077 \
/opt/project/batch_predict.py
```

Prediction results will be saved to:

```
predictions/
```

Output files are stored in **Parquet format**.

---

## Example Spark Job Output

Spark UI shows completed distributed jobs:

* Churn Prediction Model (Training)
* Batch Prediction

Workers execute tasks in parallel across the cluster.

---

## Results

The model predicts churn probability for each telecom user.

Output columns:

```
user_id
prediction
probability
```

Prediction files are stored in:

```
predictions/part-*.snappy.parquet
```

---

## Key Features

* Distributed data processing with Apache Spark
* Containerized cluster deployment using Docker
* Machine learning pipeline with Spark ML
* Batch prediction workflow
* Production-style architecture

---

## Future Improvements

* Real-time streaming predictions using Spark Streaming
* Integration with Kafka for event-driven data
* Model monitoring with MLflow
* REST API for churn prediction service

---

## Author

Manjula Srinivasan

---

## License

This project is for educational and research purposes.
