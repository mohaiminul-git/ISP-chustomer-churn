import pandas as pd

from src.data_clean import data_cleaning

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


def test_id_column_is_dropped():
    result = data_cleaning(make_df([VALID_ROW]))
    assert "id" not in result.columns


def test_column_is_renamed():
    result = data_cleaning(make_df([VALID_ROW]))
    assert "remaining_contract" in result.columns
    assert "reamining_contract" not in result.columns


def test_duplicate_rows_are_dropped():
    row1 = {**VALID_ROW, "id": 1}
    row2 = {**VALID_ROW, "id": 2}  # identical except id, which gets dropped first
    result = data_cleaning(make_df([row1, row2]))
    assert len(result) == 1


def test_missing_remaining_contract_is_flagged_and_imputed():
    row_missing = {**VALID_ROW, "id": 1, "reamining_contract": None}
    row_present = {**VALID_ROW, "id": 2, "reamining_contract": 0.8}
    result = data_cleaning(make_df([row_missing, row_present]))

    assert result.iloc[0]["missing_remaining_contract"] == 1
    assert result.iloc[1]["missing_remaining_contract"] == 0
    assert result["remaining_contract"].isna().sum() == 0
    # only one non-null value (0.8) exists across both rows, so that's the median used to fill
    assert result.iloc[0]["remaining_contract"] == 0.8


def test_negative_values_are_clipped_to_zero():
    row_negative = {**VALID_ROW, "id": 1, "subscription_age": -0.02}
    row_normal = {**VALID_ROW, "id": 2, "subscription_age": 3.0}
    result = data_cleaning(make_df([row_negative, row_normal]))

    assert (result["subscription_age"] >= 0).all()
    assert result.iloc[0]["subscription_age"] == 0
