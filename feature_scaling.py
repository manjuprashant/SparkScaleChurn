from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features"
)

scaled_data = scaler.fit(final_df).transform(final_df)