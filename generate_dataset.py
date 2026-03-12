import csv
import random

rows = 1000000

with open("data/telecom_data.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "user_id",
        "call_duration",
        "data_usage",
        "complaints",
        "billing_amount",
        "churn"
    ])

    for i in range(1, rows + 1):
        user_id = 100000 + i
        call_duration = random.randint(10, 1000)
        data_usage = round(random.uniform(0.1, 15.0), 2)
        complaints = random.randint(0, 5)
        billing_amount = random.randint(10, 150)

        # churn logic
        churn = 1 if complaints >= 3 or billing_amount < 30 else 0

        writer.writerow([
            user_id,
            call_duration,
            data_usage,
            complaints,
            billing_amount,
            churn
        ])

print("1,000,000 telecom records generated successfully")