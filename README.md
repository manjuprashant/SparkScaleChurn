
# SparkScaleChurn

## Project Overview
**SparkScaleChurn** is a scalable churn prediction project built using **Apache Spark** and **PySpark MLlib**. The project processes customer data, trains machine learning models, and predicts the likelihood of customer churn. It is designed to handle large datasets efficiently with Spark’s distributed computing capabilities.

The project includes:
- Data preprocessing and feature engineering.
- Multiple ML pipelines: Decision Tree, Random Forest, Gradient Boosted Trees, and Logistic Regression.
- Model evaluation and metrics tracking.
- Storage of processed data and trained models for future inference.

---

## Folder Structure


SparkScaleChurn/
│
├── data/ # Raw input datasets
├── processed/ # Processed output datasets
├── models/ # Trained ML models and pipelines
├── backup/ # Backup of original scripts/data
├── BinaryClassificationEvaluator.py
├── batch_predict.py
├── data_processing.py
├── evaluate.py
├── evaluation_metrics.txt
├── evaluation_report_churn.pdf
├── generate_dataset.py
├── train_model.py
├── week3_modelling_with_metrics.py
└── README.md # Project documentation


---

## Prerequisites
- Python 3.14+
- Apache Spark 3.5+
- PySpark
- pandas, numpy, matplotlib, seaborn (for EDA and visualization)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/manjuprashant/SparkScaleChurn.git
cd SparkScaleChurn

Create a virtual environment and install dependencies:

python -m venv venv
.\venv\Scripts\activate       # Windows
pip install -r requirements.txt

Make sure Spark is installed and available in your PATH.

Usage

Data Generation & Processing

python generate_dataset.py       # Generate synthetic dataset (if applicable)
python data_processing.py        # Clean and process raw data

Processed data will be saved in processed/.

Model Training

python train_model.py

Trained models are saved in the models/ folder.

Batch Predictions

python batch_predict.py

Predictions are stored in output/ folder.

Evaluation

python evaluate.py

Generates evaluation metrics and reports (evaluation_metrics.txt, evaluation_report_churn.pdf).

Models Included

Decision Tree Classifier

Random Forest Classifier

Gradient Boosted Trees (GBT) Classifier

Logistic Regression

All models are stored under models/ and can be reloaded for inference.

Contributing

Fork the repo

Create a feature branch (git checkout -b feature-name)

Commit your changes (git commit -m 'Add feature')

Push to the branch (git push origin feature-name)

Open a Pull Request

License

This project is licensed under the MIT License.

Contact

Manjula Srinivasan
GitHub: https://github.com/manjuprashant
