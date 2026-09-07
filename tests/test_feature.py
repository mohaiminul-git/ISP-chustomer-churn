import pandas as pd

from src.features import generate_features

VALID_ROW = {
    "id": 1,
    "is_tv_subscriber": 1,
    "is_movie_package_subscriber": 0,
    "subscription_age": 2.5,
    "bill_avg": 20,
    "reamining_contract": 0.5,
    "service_failure_count": 0,
    "download_avg": 10.0,
    "upload_avg": 2.0,
    "download_over_limit": 0,
    "churn": 0,
}


def make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_featureGeneration():
    df= make_df([VALID_ROW])
    featured_df= generate_features(df)
    assert "num_services_subscribed" in featured_df.columns
    assert "cost_per_usage" in featured_df.columns

def test_cose_perUsage_value():
    row={**VALID_ROW,"bill_avg": 0,"download_avg": 0,"upload_avg": 0}
    df= make_df([row])
    featured_df= generate_features(df)
    assert featured_df["cost_per_usage"].iloc[0]==0
    
def test_cose_perUsage_value_withNumber():
    row={**VALID_ROW,"bill_avg": 1,"download_avg": 1,"upload_avg": 1}
    df= make_df([row])
    featured_df= generate_features(df)
    assert featured_df["cost_per_usage"].iloc[0]==0.5
    assert featured_df["num_services_subscribed"].iloc[0]==1
    
    