import pandas as pd

TIME_COL = "MonthYear"
TARGET_COL = "PatientVisits_Sum"
AGE_COL = "AgeGroup"
GENDER_COL = "Gender"
DRUG_COL = "DrugName"
COMPANY_COL = "CompanyName"
TYPE_COL = "type"
SPEC_COL = "Category_Grouped"

COVID_CUTOFF_DATE = "2021-01-01"
TOP_PERCENTILE = 75


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    df["MonthYear_dt"] = pd.to_datetime(
        df[TIME_COL], errors="coerce"
    )
    df = df.dropna(subset=["MonthYear_dt"])

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL] > 0]

    df = df[df["MonthYear_dt"] >= COVID_CUTOFF_DATE]

    if df.empty:
        raise ValueError("No data left after date/target filtering")

    df_agg = (
        df.groupby([DRUG_COL, COMPANY_COL, SPEC_COL, AGE_COL, GENDER_COL, TYPE_COL])
        .agg(Total_Volume=(TARGET_COL, "sum"))
        .reset_index()
    )

    df_agg["Volume_Percentile"] = (
        df_agg.groupby(SPEC_COL)["Total_Volume"].rank(pct=True)
    )

    df_agg["Target"] = (
        df_agg["Volume_Percentile"] >= (TOP_PERCENTILE / 100)
    ).astype(int)

    if df_agg["Target"].nunique() < 2:
        raise ValueError("Not enough class diversity for training")

    return df_agg
