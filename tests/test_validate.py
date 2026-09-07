import pandas as pd
import pytest

from src.validate import validate_data

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


def make_df(**overrides) -> pd.DataFrame:
    row = {**VALID_ROW, **overrides}
    return pd.DataFrame([row])


def test_valid_dataframe_returns_no_warnings():
    df = make_df()
    assert validate_data(df) == []


def test_missing_column_raises():
    df = make_df().drop(columns=["bill_avg"])
    with pytest.raises(ValueError, match="Missing expectecd columns"):
        validate_data(df)


def test_churn_outside_0_1_raises():
    df = make_df(churn=2)
    with pytest.raises(ValueError, match="Target variable"):
        validate_data(df)


def test_negative_value_warns_but_does_not_raise():
    df = make_df(subscription_age=-0.02)
    warnings = validate_data(df)
    assert len(warnings) == 1
    assert "subscription_age" in warnings[0]
    assert "smaller than 0" in warnings[0]


def test_negative_value_reports_every_offending_column():
    df = make_df(subscription_age=-1, bill_avg=-5)
    warnings = validate_data(df)
    assert len(warnings) == 2
