from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

PREDICTION_COUNTER = Counter(
    "churn_predictions_total",
    "Total number of churn predictions made",
    ["endpoint", "predicted_class"],
)

PREDICTION_LATENCY = Histogram(
    "churn_prediction_latency_seconds",
    "Time spent generating a churn prediction",
    ["endpoint"],
)