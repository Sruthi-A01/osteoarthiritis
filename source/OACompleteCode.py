#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import log
import re
import nltk
from nltk.corpus import stopwords
from statsmodels.tsa.seasonal import seasonal_decompose
from itertools import combinations
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from pandas.plotting import scatter_matrix
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_SM = True
except Exception:
    HAS_SM = False


# In[2]:


df=pd.read_excel("/content/Transformed_Dataset.xlsx")


# In[3]:


print("\n Shape (rows × columns):")
print(df.shape)


# In[4]:


print("\n Data types:")
print(df.dtypes)


# In[5]:


print("\n Memory usage (MB):")
mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
print(f"{mem_mb:.4f} MB")


# In[6]:


print("\n First 5 rows (head):")
display(df.head())


# In[7]:


print("\n Last 5 rows (tail):")
display(df.tail())


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


missing_df = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percent": (df.isna().mean() * 100).round(2)
}).sort_values("missing_percent", ascending=False)

print(" Missing Values Summary:")
display(missing_df)


# In[11]:


duplicate_count = df.duplicated().sum()
print(f"\n Duplicate Rows: {duplicate_count}")
if duplicate_count > 0:
    print("\nSample duplicate rows:")
    display(df[df.duplicated()].head())


# In[12]:


numeric_like_cols = []
for col in df.columns:
    if df[col].dtype == "object":
        try:
            pd.to_numeric(df[col].dropna().sample(min(50, df[col].notna().sum())), errors="raise")
            numeric_like_cols.append(col)
        except:
            pass
print(numeric_like_cols if numeric_like_cols else "None found")


# In[13]:


constant_cols = []
near_constant_cols = []

for col in df.columns:
    top_freq = df[col].value_counts(normalize=True, dropna=False).values[0]
    if top_freq == 1:
        constant_cols.append(col)
    elif top_freq >= 0.995:
        near_constant_cols.append(col)

print("\n Constant columns (same value in 100% rows):")
print(constant_cols if constant_cols else "None")

print("\n Near-constant columns (>99.5% same value):")
print(near_constant_cols if near_constant_cols else "None")


# In[14]:


invalid_numeric_report = []

for col in df.select_dtypes(include=[np.number]).columns:
    negatives = (df[col] < 0).sum()
    zeros     = (df[col] == 0).sum()
    if negatives > 0 or zeros > 0:
        invalid_numeric_report.append({
            "column": col,
            "negative_count": int(negatives),
            "zero_count": int(zeros)
        })

invalid_numeric_df = pd.DataFrame(invalid_numeric_report)

print("\n Invalid Numeric Value Check (negative/zero counts):")
display(invalid_numeric_df if not invalid_numeric_df.empty else pd.DataFrame({"status":["No invalid numeric values found"]}))


# In[15]:


date_columns = []
for col in df.columns:
    if df[col].dtype == "object":
        try:
            pd.to_datetime(df[col].dropna().iloc[:50], errors="raise")
            date_columns.append(col)
        except:
            pass

print("\n Detected Date-like columns:", date_columns if date_columns else "None")

invalid_dates_report = {}

for col in date_columns:
    converted = pd.to_datetime(df[col], errors="coerce")
    invalids = converted.isna() & df[col].notna()
    if invalids.sum() > 0:
        invalid_dates_report[col] = int(invalids.sum())

print("\n Invalid Date Values:")
print(invalid_dates_report if invalid_dates_report else "No invalid dates found")


# In[16]:


numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

print("Numeric columns detected:", numeric_cols)
print("Categorical columns detected:", categorical_cols)


# In[17]:


for col in categorical_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.title()
    )


# In[18]:


for col in categorical_cols:
    df[col] = df[col].replace({
        "Other ": "Other",
        "OTHER": "Other"
    })


# In[19]:


if numeric_cols:
    print("\nDESCRIPTIVE STATISTICS — NUMERIC COLUMNS")
    numeric_desc = df[numeric_cols].describe().T
    numeric_desc["skew"] = df[numeric_cols].skew()
    numeric_desc["kurtosis"] = df[numeric_cols].kurtosis()
    display(numeric_desc)
else:
    print("\n No numeric columns found.")


# Results from this descriptive statistics tells that having max value as 4717 can tell that some patients are keeping coming again and again defining the chronic care. The average is much higher than the median where median is 4 (defining the person comes 4 times) and mean is 51(average is 51 visits) very small group of patients has huge number of visists and they raise the average defining the hravy tail distribution. Standard deviation is 194 tells that is huge as visit counts are spread out and there is big variation from person to person and some visits are rarely some visit extremely often which is normal in helathcare w.r.t in chronic vs acute patients.
# 
# Skewness is 8.58 defining the high skenwss where if skewness is close to 0 id balanced and skew >1 is heavily right skewed.Not symmetric at all.
# 
# Kurtosis had value as 100 defining the extreme outliers and kurtosis tells how many outliers exist and the kurtosos around value 3 is normal distribution and around 10 is very heavy tails and we have 100 that defines many extreme values.
# 
# In this case the regualr statistical models will not work like Linear regression, lOGISTIC REGRESSION, SVM, K-means clustering and tree models can work well as they dont care about the skewness or outliers.

# In[20]:


if categorical_cols:
    print("\nVALUE COUNTS — CATEGORICAL COLUMNS")

    for c in categorical_cols:
        print(f"\nColumn: {c}")
        vc = df[c].value_counts(dropna=False)
        vc_pct = (vc / len(df) * 100).round(2)
        vc_table = pd.DataFrame({"count": vc, "percent": vc_pct})
        display(vc_table.head(20))
else:
    print("\n No categorical columns found.")


# Dataset is Dominated by two specialites where primary care is 40% and orthopoedic wiht 34% followed with other things this tells that there is class imbalance mentioning that any model prediction can lead to focus more on Primary care+Ortho as some patterns may llok more stable in major groups but unstable in minor groups due to fewer samples. Time-series stability is excellent as model training will not suffer from month-level bais and can do the trend analysis and seasonality detection.
# 
# High utilizationpatterns are mostly come from older populations and age will be a strong predictor of visit counts, When doing mOdeling for younger groups beware of data sparsity - Data Sparsity refers to the condition where a large percentage of data within a dataset is missing or is set to zero.
# 
# 
# Gender distribution is balanced and can be used as a clean feature.
# 
# For CompanyName we can see the long tail distribution of many smaller companies so models based on this need handling of rare levels like doing the grouping rare companies into "Other", target encoding, count encoding, class weights if modelling companyname is target
# 
# DrugName also contains the Long-tail of low frequency drugs leading t long-tail distribution as many drugs are rare. This shows the highly imbalanced feature if drug is considered as target then imbalance must be handled. If drugname is considered as input then do one-hot encoding will be sparse and consider target encoding or frequency encoding.
# 
# Type is strong categorical feature no imbalance issues.

# In[21]:


numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
print("Numeric columns:", numeric_cols)


# In[22]:


for col in numeric_cols:
    plt.figure(figsize=(10,5))
    sns.histplot(df[col].dropna(), kde=True, bins=40, color='blue')
    plt.title(f"Histogram & KDE for {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


# The data is NOT normal, not symmetric, not smooth.
# Because 99% of data is between 1 and ~100, and only 0.1% go beyond that.
# The extreme values stretch the X-axis so much that everything else becomes compressed.
# 
# 

# In[23]:


stats_summary = []

for col in numeric_cols:
    s = df[col].dropna()
    stats_summary.append({
        "column": col,
        "mean": s.mean(),
        "median": s.median(),
        "mode": s.mode().iloc[0] if not s.mode().empty else np.nan,
        "variance": s.var(),
        "std_dev": s.std(),
        "skewness": s.skew(),
        "kurtosis": s.kurtosis()
    })

stats_df = pd.DataFrame(stats_summary)
print("\n Descriptive Statistics:")
display(stats_df)


# In[24]:


outlier_report = []

for col in numeric_cols:
    s = df[col].dropna()

    z = (s - s.mean()) / s.std()
    z_outliers = (np.abs(z) > 3).sum()

    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    iqr_outliers = ((s < Q1 - 1.5 * IQR) | (s > Q3 + 1.5 * IQR)).sum()

    outlier_report.append({
        "column": col,
        "zscore_outliers": int(z_outliers),
        "iqr_outliers": int(iqr_outliers),
        "total_rows": len(s)
    })

outliers_df = pd.DataFrame(outlier_report)
print("\n Outlier Detection Summary:")
display(outliers_df)


# Z-score works only for normal-like data → your data is NOT normal → Z-score missed most outliers.
# IQR works for skewed data → caught a LOT more outliers (correct).
# 
# They are not mistakes — they are high-utilization patients, a real healthcare pattern.
# Don’t remove them blindly.
# 
# Instead:
# transform
# scale
# or use tree-based models (which handle them well)

# In[25]:


missing_summary = pd.DataFrame({
    "missing_count": df[numeric_cols].isna().sum(),
    "missing_percent": (df[numeric_cols].isna().mean() * 100).round(2)
})

print("\n Missing Value Summary:")
display(missing_summary)


# In[26]:


cat_cols = [c for c in df.columns if df[c].dtype == "object"]
print("Categorical Columns:", cat_cols)

categorical_report = []


# In[27]:


for col in cat_cols:

    print(f"ANALYSIS FOR COLUMN: {col}")

    s = df[col].astype(str)

    counts = s.value_counts(dropna=False)
    pct = (counts / len(s) * 100).round(2)

    freq_table = pd.DataFrame({"count": counts, "percent": pct})
    print("Top Categories:")
    display(freq_table.head(20))


    plt.figure(figsize=(12,5))
    sns.barplot(x=freq_table.head(20).index, y=freq_table.head(20)["count"], palette="Blues_d")
    plt.xticks(rotation=75)
    plt.title(f"Top 20 Categories in {col}")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.4)
    plt.show()

    cardinality = s.nunique()
    print(f"Cardinality (unique categories): {cardinality}")

    rare_5 = freq_table[freq_table["percent"] < 5]
    rare_1 = freq_table[freq_table["percent"] < 1]

    print("\nCategories < 5%:")
    display(rare_5)

    print("Categories < 1%:")
    display(rare_1)

    # IMBALANCE CHECK (dominance)
    dominant = pct.max()
    dominant_category = pct.idxmax()

    print(f" Dominant category: '{dominant_category}' with {dominant}% share")

    if dominant > 50:
        print(" Severe imbalance detected (dominant > 50%)")
    elif dominant > 30:
        print(" Moderate imbalance detected (dominant > 30%)")
    else:
        print(" No major imbalance")

    probs = pct / 100
    entropy = -(probs * np.log2(probs)).sum()

    print(f"Entropy: {entropy:.4f} bits")
    categorical_report.append({
        "column": col,
        "cardinality": cardinality,
        "dominant_category": dominant_category,
        "dominant_percent": dominant,
        "entropy": entropy,
        "rare_categories_<5%": len(rare_5),
        "rare_categories_<1%": len(rare_1)
    })


# 1. Category_Grouped: Primary Care 40.23% dominates , Orthopedic is Strong and followed by Others and Entropy is 1.80 bits defining that few categories hold most of the mass. It is having moderate skew, but use stratified splits if category_grouped in the target or key segment. If adding as feature no need to rebalance, but used as target consider class weights or macro metrics to avoid bias.
# 
# 2. Month Year: Has 72 unique months, entropy has 6.16 bits (high, for many almost-equal bins) and it is have great coverage for time modelling and low risk of leakage and is very ideal for time-series splits like train on earlier months and validate on later.
# Extract the Year,month, Quarter,and Time Index, for forecasting keep chronologival train/val/test splits.
# 
# 3. Age Group: Entropy is 2.54 bits whihc is decently distributed.
# Balanced enough for strtification no rebalancing needed. Good efffects with interaction effects with Category_Grouped,Gender, DrugName. Consider ordering (ordinal encoding )if modelling assumes order.
# 
# 4. Gender: Entropy 1.03 bits low (beacuse one class dominates and one is tiny)
# Severe imbalance threshold corssed(>50%) and unspecified is tiny where models can overfit it ftreates as full class.
# Feature Use: Keep all three and consider target encoding and let the model learn.
# If Gender is the target, use class weights or resamplimg.
# 
# 5. CompanyName: Head is diversified where Pfizer is top share, but most companies are rare(<1% many, long tail). Entropy 5.40 bits -high but with a heavy tail. Classic high-cardinality problem so dont go with one-hot as it creates many near-empty columns and harming the generalization.
# Use grequency encoding or target mean encoding with K-fold smoothing to avoid leakage.
# Optional Step : Group rare compnaies into Other for reporting but for modeling prefer encodings above.
# 
# 6. DrugName: Multiple drugs in 9-11% range, then a long tail of rares(<1%). Entropy is 4.42 bits (diverse, long tail)
# High-cardinalty challenge. Prefer frequency/target encoding or hashing for linear models.
# Consider buikding therapeutic class features (map drug ->class) to reduce cardinality with domain signal.
# 
# 7.Type: Generic dominates (51.78%) Entropy 1.71 bits, severe imbalance due to Generic. If type is feature imbalance is fine and it becomes the target use class weights and macro metrics.
# 

# In[28]:


target = "PatientVisits_Sum"

plt.figure(figsize=(10,5))
sns.histplot(df[target], kde=True, bins=40, color='blue')
plt.title("Histogram + KDE of Target Variable (PatientVisits_Sum)")
plt.show()

plt.figure(figsize=(8,4))
sns.boxplot(x=df[target])
plt.title("Boxplot for PatientVisits_Sum")
plt.show()

print("\nTarget Variable Basic Stats:")
display(df[target].describe())

print("\nSkewness:", df[target].skew())
print("Kurtosis:", df[target].kurtosis())

df["MonthYear_dt"] = pd.to_datetime(df["MonthYear"])

monthly = df.groupby("MonthYear_dt")[target].sum().sort_index()

plt.figure(figsize=(12,5))
plt.plot(monthly.index, monthly.values)
plt.title("Monthly Total Patient Visits Over Time")
plt.xlabel("Month")
plt.ylabel("Total Visits")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12,5))
plt.plot(monthly.index, monthly.values, label="Actual")
plt.plot(monthly.index, monthly.rolling(3).mean(), label="3-Month Rolling Avg", linewidth=3)
plt.legend()
plt.title("Trend with Rolling Average")
plt.xlabel("Month")
plt.ylabel("Total Visits")
plt.show()

decomp = seasonal_decompose(monthly, model='additive', period=12)
decomp.plot()
plt.show()


# Histogram+KDE Tells about the extreme right skewed distribution
# 
# Boxplottells that there are many outliers where 50% of patients have 4 or fewer visits as some have 500,1000,2000,4000+visists outliers tells the real extreme behaviour and healthcare datasets follow "power-law behaviour" and also defining that few individuals consume a huge amout of resources. Z-Score is not prefect for skewed data and IQR Outlier Detection works better for skewed data.
# 
# Going with the skewness 0 is perfectly symmetrical, 1 is moderately skewed, 2 is strongly skewed, a skewness above 2 means some log transofrmation is recommended.
# Should go with Log Transformation, Robust Scaling, Winsorization.
# 
# Kurtosis defines the data distribution, describing the degree to which data points are outliers compared to a normal distribution.
# Normal distribution follows around 3 but out data has 100 defining heavy tail distribution defining the extreme values dominate the dataset, Variance is not stable.
# 
# The line plot tells the trends like this
# The line plot shows:
# Sharp dips (COVID shock around 2020)
# A recovery wave in 2021
# Peak volumes in 2021–2022
# A slow decline or stabilization in 2023–2024
# A slight uptick again near 2025
# 
# Rolling Average : shows the same results more like and has a meaningful structure, time-series modeling(Prphet, ARIMA ,LSTM , SARIMA, Seasonal Decompostition, Markov Models) is valid and seasonlaity is strong and stable.
# 
# 
# Trend: Tells that Increases until 2022 and declines slowly afterward, tells about the real-world clinical behaviour (spike and stabilization)
# 
# Seasonal Component: Good month oscillations, very helpful for forecasting
# 
# Residuals:They define what are left with trends and seasonlaities, They represent the random noise or unpredictable variation in the data that the trend and seasonality components couldn't explain and here in this residuals are mostly centered around zero and big differences( around pandemic) and the decomposition is valid.

# Stability of Target Varibale : Helps to check the target behaves consistently over time and across groups, telling whether it drifts, shifts,or changes patterns. Important for predicitve modeling,forecasting,fairness analysis,feature engineering, understanding the real-world behaviour

# In[29]:


TARGET_COL = "PatientVisits_Sum"
CAT_COLS = ["Category_Grouped", "MonthYear", "AgeGroup", "Gender",
            "CompanyName", "DrugName", "type"]
TOP_K = 10
TOP_K_LINES = 6
ROLL_WINDOW = 3
SPIKE_PCT = 0.25


df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")


if "type" in df.columns:
    df["type"] = (df["type"]
                  .astype(str)
                  .str.strip()
                  .str.title()
                  .replace({"Others": "Other"}))


if "MonthYear" in df.columns:
    df["MonthYear_dt"] = pd.to_datetime(df["MonthYear"], errors="coerce")

    df["MonthStart"] = df["MonthYear_dt"].dt.to_period("M").dt.to_timestamp()
else:

    df["MonthStart"] = pd.NaT

df = df[~df[TARGET_COL].isna()].copy()

print(f"Rows after target cleaning: {len(df):,}")


if df["MonthStart"].notna().any():
    monthly = (df.groupby("MonthStart", as_index=False)[TARGET_COL]
                 .agg(total="sum", mean="mean", median="median", n="count"))
    monthly["pct_change_total"] = monthly["total"].pct_change()
    monthly["rolling_total"] = monthly["total"].rolling(ROLL_WINDOW, min_periods=1).mean()
    spikes = monthly.loc[monthly["pct_change_total"].abs() > SPIKE_PCT,
                         ["MonthStart", "total", "pct_change_total"]]

    display(monthly.head(12))
    print("\n Potential monthly spikes (>|25%| change):")
    display(spikes)


    plt.figure(figsize=(12,4.5))
    plt.plot(monthly["MonthStart"], monthly["total"], label="Total visits")
    plt.plot(monthly["MonthStart"], monthly["rolling_total"],
             label=f"{ROLL_WINDOW}-month rolling", linewidth=2)
    plt.title("Monthly Total Patient Visits (with rolling average)")
    plt.xlabel("Month"); plt.ylabel("Total Visits"); plt.legend(); plt.tight_layout(); plt.show()


    plt.figure(figsize=(12,3.8))
    plt.plot(monthly["MonthStart"], monthly["pct_change_total"]*100)
    plt.axhline(SPIKE_PCT*100, color="r", linestyle="--", linewidth=1)
    plt.axhline(-SPIKE_PCT*100, color="r", linestyle="--", linewidth=1)
    plt.title("Month-to-Month % Change in Total Visits (spike bands)")
    plt.xlabel("Month"); plt.ylabel("% change"); plt.tight_layout(); plt.show()


def top_categories(df, col, k=TOP_K):
    freq = (df[col].astype(str)
              .value_counts(dropna=False)
              .rename("count")
              .to_frame())
    freq["percent"] = (freq["count"] / len(df) * 100).round(2)
    return freq.head(k)

def category_summary(df, col):

    agg = (df.groupby(col, dropna=False)[TARGET_COL]
             .agg(total="sum", mean="mean", median="median", n="count")
             .sort_values("total", ascending=False))
    return agg

def category_time_series(df, col, top_k=TOP_K_LINES):

    top_list = (df.groupby(col)[TARGET_COL].sum()
                  .sort_values(ascending=False)
                  .head(top_k).index.tolist())

    ts = (df[df[col].isin(top_list)]
            .groupby(["MonthStart", col])[TARGET_COL].sum()
            .reset_index())
    return ts, top_list

def plot_bar_top(df, col, k=TOP_K):
    freq = top_categories(df, col, k)
    plt.figure(figsize=(10,4))
    sns.barplot(x=freq.index.astype(str), y=freq["count"], color="#4C78A8")
    plt.title(f"Top {k} categories in {col} (counts)")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.show()
    return freq

def volatility_by_share(df, col):
    if df["MonthStart"].isna().all():
        return pd.DataFrame(columns=[col, "share_std"])
    month_tot = df.groupby("MonthStart")[TARGET_COL].sum().rename("m_total")
    cat_month = (df.groupby(["MonthStart", col])[TARGET_COL]
                   .sum()
                   .reset_index()
                   .merge(month_tot, on="MonthStart", how="left"))
    cat_month["share"] = cat_month[TARGET_COL] / cat_month["m_total"]
    vol = (cat_month.groupby(col)["share"].std()
             .sort_values(ascending=False)
             .rename("share_std")
             .to_frame())
    return vol

for col in CAT_COLS:
    if col not in df.columns:
        print(f"Skipping {col} (not in dataframe)");
        continue

    print("\n"+"="*80)
    print(f" Column: {col}")
    freq = plot_bar_top(df, col, k=TOP_K)
    display(freq)

    agg = category_summary(df, col)
    display(agg.head(10))

    if df["MonthStart"].notna().any():
        vol = volatility_by_share(df, col)
        print("Volatility of monthly share (std): top 10")
        display(vol.head(10))
        ts, top_list = category_time_series(df, col, top_k=TOP_K_LINES)
        if not ts.empty:
            plt.figure(figsize=(12,4.5))
            for c in top_list:
                sub = ts[ts[col]==c].sort_values("MonthStart")
                plt.plot(sub["MonthStart"], sub[TARGET_COL], label=str(c))
            plt.title(f"Monthly totals of top {len(top_list)} {col} categories")
            plt.xlabel("Month"); plt.ylabel(TARGET_COL); plt.legend(ncol=2, fontsize=9)
            plt.tight_layout(); plt.show()


# Trend and Stability Over Time (Month-to-Month Behaviour)
# Follows the normal pattern and stable before and huge drop in April 2020 and we see the immediate rebound in May-June 2020 and further strong stable growth period (2021-earlt 2023) and recent decline in 2024-2025.
# 
# With the rolling averagae we got to know more about the target is not stationary and has drift over the years. Target VARIABLE HAS TEMPORAL INSTABILITY. It has trend+seasonality+shock events.
# 
# By each category:
# Volatility (How much they share changes month-to-month and from the Category_Grouped we see that Primary care and orthopedic flucutate the most over time and others are extremely stable.
# 
# By Age group: Older groups flucutate more and younger groups are extremely stable (low usage)
# 
# By Gender: Women have higher visit density with the rheumatology tendencies. Male and Female usage drifts similarly over time and no major gender shocks.
# 
# 
# By MonthYear: The chart shows nearly even distribution and each month has roughly same no of records and total visits are meaningful here.
# 
# By CompanyName: the market is not dominated by one or two companies.No single company crosses even 10%share and the company level distribution is fairly balanced. Multiple manufactures contribute to visits. Good for modelling and no extreme imbalance.
# Comapnies with high vloumes are not always the companies with the highest visit counts.and where the Bristol-M and Pfizer clearly dominate the visist volume not row count.
# Volatility tells how much each company monthly market share jumps around.
# 
# By DrugName: Drug Distribution is concentrated (more than company) and few drugs dominate prescribing patterns. High row count is not equal to high visit count.
# 
# ByType: The category Distribution the variable shows the strong imbalance as Generic goes for half of all rows and branded generic contribute to much higher mean visits. Branded Generic has the highest mean visits and Other type is very low intensity.
# 
# Going with the category effects AgeGroup, CompanyName, DrugName, Type and helpful for Regression, Clustering, Forecasting and Market Segmentation.
# 
# Volatility helps to decide which features are stable, reliable predictors and which features are unstable, need smoothing or lag features.
# 
# Switching Behaviour: CompanyName+DrugName volatility can help for transition Matrices, Markov Chains, Switching Probability models.
# 
# Across categories:
# AgeGroup → stable, predictable patterns
# Gender → slight imbalance but trend consistent
# Category_Grouped → Primary Care + Orthopedic dominate
# CompanyName → Bristol-Myers and Pfizer drive visit volume
# DrugName → Kenalog + Depo-Medrol are the largest contributors
# Type → Generic dominates volume; Branded Generic dominates intensity
# Volatility varies widely, indicating different levels of stability across categories.
# 

# In[30]:


TARGET = "PatientVisits_Sum"
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
categorical_cols = [c for c in df.columns if c not in numeric_cols]

for col in categorical_cols.copy():
    try:
        pd.to_datetime(df[col].dropna().head(20))
        categorical_cols.remove(col)
    except:
        pass

print("Numeric:", numeric_cols)
print("Categorical:", categorical_cols)


# In[31]:


TOP_K = 10

for col in categorical_cols:
    d = df[[col, TARGET]].dropna()
    if d[col].nunique() < 2:
        continue
    used_subset = False
    if d[col].nunique() > TOP_K:
        top_levels = (
            d.groupby(col)[TARGET]
              .sum()
              .sort_values(ascending=False)
              .head(TOP_K)
              .index
        )
        d = d[d[col].isin(top_levels)].copy()
        used_subset = True

    order = d.groupby(col)[TARGET].median().sort_values().index
    d[col] = pd.Categorical(d[col], categories=order, ordered=True)

    plt.figure(figsize=(8, 4))
    d.boxplot(column=TARGET, by=col, vert=False)
    ttl = f"{TARGET} by {col}"
    if used_subset:
        ttl += f" (top {TOP_K} by total {TARGET})"
    plt.title(ttl)
    plt.suptitle("")
    plt.xlabel(TARGET)
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()
    try:
        model = smf.ols(f"{TARGET} ~ C({col})", data=d).fit(cov_type="HC3")
        a = anova_lm(model, typ=2)
        print(f"\nANOVA for {col}" + (" (top-k subset)" if used_subset else "") + "\n")
        print(a)

    except Exception as e:
        print(f"ANOVA failed for {col}: {e}\n")


# Category_Grouped -> PatientVisits_Sum
# 
# Boxplot defining that Primary Care and Orthopedic have the Higher Medians and massive outliers and very wide distribution other and onco have low median values and much tighter distribution. Anova tells that some categories deals with more patienrs while others handle fewer and the category strongly influences patient volumes.
# 
# 
# AgeGroup -> PatientVisits_Sum
# 
# Age group has major impact on total patient visits.
# 
# 
# Gender -> Patientvisits
# Female and Male have large similar distributions and unspecified has almost no volume gender still influences visit counts, but the effect is smaller compared to age or category.
# 
# CompanyName->PatientVisits_Sum
# High-Cardinality categorical feature, some drug manufactures dominate the patient volume. CompanyName is very strong differentaitor for patient volume.
# 
# DrugName->patientvisits
# Less used drugs ahve tight clusters
# srugname is also highly influential.
# 
# Type->patientvisits
# Branded Generics and Brands show:
# Very high outliers
# Higher medians
# Generic and Other:
# Lower median visits
# Tighter spread
# 
# ComapnyName: Some companies are associated with extremely high patient visit counts. CompanyName is a strong explanatory variable for patient visit counts.
# 
# DrugName: Different drugs have drastically different visit distributions.The variability across drugs is even stronger than across companies.
# DrugName has more explanatory power than CompanyName
# 
# the type of drug strongly affects total visit counts.
# as all anove p-values are 0 Category, AgeGroup, Gender, CompanyName, DrugName, and type all show significant differences in how many visits they receive.”
# 
# 
# 
# CompanyName and DrugName requires target encoding, Frequency Encoding and Catboost internal handling
# 
# Anova tells that categories are meaningful and supports Segmentation analysis
# Stratified modeling
# Interaction modeling later (Category × AgeGroup)

# In[32]:


for c1, c2 in combinations(categorical_cols, 2):
    d = df[[c1, c2]].dropna()
    if d[c1].nunique() < 2 or d[c2].nunique() < 2:
        continue

    ct = pd.crosstab(d[c1], d[c2])
    chi2, p, dof, exp = stats.chi2_contingency(ct)

    print(f"{c1} ↔ {c2} | p-value={p:.4f}, dof={dof}")
    if p < 0.05:
        print("  → Significant association")
    print()


# 1. All categorical features are not independent. They are strongly correlated or associated.
# 2. Many category features carry overlapping signal
# 3. There is high multi-collinearity risk espicially when using Logistic/Linear regression, GLM , ANOVA
# 4. Time-based models like CatBoost,XGBoost and RF will handle this multicollinearity naturally.
# 5. For regression models, we have to check VIF(Variance Inflation Factor) - how much a predictor (independent variable) is correlated with other predictors. Also, Many categorical features may need dimensionality reduction, target encoding and grouping low-frequency categories.

# In[33]:


def cramers_v(confusion_matrix):
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1),(rcorr-1)))

import scipy.stats as ss

cat_pairs = []
for c1 in cat_cols:
    for c2 in cat_cols:
        if c1 >= c2:
            continue
        table = pd.crosstab(df[c1], df[c2])
        v = cramers_v(table.values)
        cat_pairs.append([c1, c2, v])

cv_df = pd.DataFrame(cat_pairs, columns=["Feature1","Feature2","CramersV"])
display(cv_df.sort_values("CramersV", ascending=False).head(20))


plt.figure(figsize=(12,6))
sns.heatmap(
    pd.crosstab(df[cat_cols[0]], df[cat_cols[1]]),
    cmap="Blues"
)
plt.title(f"Crosstab Heatmap: {cat_cols[0]} vs {cat_cols[1]}")
plt.show()


# In[34]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

cat_cols = ['Category_Grouped', 'MonthYear', 'AgeGroup', 'Gender', 'type',
            'CompanyName', 'DrugName']
cardinality = {col: df[col].nunique() for col in cat_cols}

print("Category cardinalities:")
print(cardinality)
MAX_CATS = 20

for c1 in cat_cols:
    for c2 in cat_cols:
        if c1 >= c2:
            continue

        if df[c1].nunique() > MAX_CATS or df[c2].nunique() > MAX_CATS:
            print(f"\nSkipping heatmap for {c1} vs {c2} (too many categories)")
            continue

        print(f"\nPlotting heatmap: {c1} vs {c2}")

        ct = pd.crosstab(df[c1], df[c2])

        plt.figure(figsize=(14,6))
        sns.heatmap(ct, cmap="Blues")
        plt.title(f"Crosstab Heatmap: {c1} vs {c2}")
        plt.show()


# In[34]:




