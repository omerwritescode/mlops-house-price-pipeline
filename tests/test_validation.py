import numpy as np

from hpp.validation import validate_frame


def test_supplied_training_data_passes(cfg, train_df):
    report = validate_frame(train_df, cfg)
    assert report["passed"], report["message"]


def test_each_incoming_batch_passes(cfg, incoming_df):
    for _, part in incoming_df.groupby("Batch"):
        assert validate_frame(part, cfg)["passed"]


def test_missing_column_fails_schema(cfg, train_df):
    report = validate_frame(train_df.drop(columns=["City_Zone"]), cfg)
    assert not report["passed"] and report["failed_checks"] == ["schema.required_columns"]


def test_excess_missing_values_fail(cfg, train_df):
    df = train_df.copy()
    df.loc[df.sample(frac=0.3, random_state=0).index, "Neighborhood_Score"] = np.nan
    report = validate_frame(df, cfg)
    assert "missing.Neighborhood_Score" in report["failed_checks"]


def test_out_of_range_and_unknown_category_fail(cfg, train_df):
    df = train_df.copy().astype({"City_Zone": object})
    df.loc[:59, "Area_sqft"] = -100
    df.loc[60:99, "City_Zone"] = "Downtown"
    report = validate_frame(df, cfg)
    assert "validity.invalid_rows" in report["failed_checks"]
    assert "range.Area_sqft" in report["violations"] and "category.City_Zone" in report["violations"]


def test_text_in_numeric_column_fails(cfg, train_df):
    df = train_df.copy().astype({"Bedrooms": object})
    df.loc[:99, "Bedrooms"] = "three"
    assert "schema.numeric_types" in validate_frame(df, cfg)["failed_checks"]


def test_missing_target_fails(cfg, train_df):
    df = train_df.copy()
    df.loc[:4, "Price_Lakhs"] = np.nan
    assert "missing.Price_Lakhs" in validate_frame(df, cfg)["failed_checks"]


def test_too_few_rows_fail(cfg, train_df):
    assert "volume.min_rows" in validate_frame(train_df.head(50), cfg)["failed_checks"]


def test_duplicate_ids_fail(cfg, train_df):
    df = train_df.copy()
    df.loc[1, "House_ID"] = df.loc[0, "House_ID"]
    assert "uniqueness.duplicate_ids" in validate_frame(df, cfg)["failed_checks"]
