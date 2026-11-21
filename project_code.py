
import pandas as pd
import numpy as np

# 1. LOAD THE DATASET

df = pd.read_csv("Hotel_Booking.csv")

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nRandom 5 rows:")
print(df.sample(min(5, len(df)), random_state=42))

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nStatistical description:")
print(df.describe(include="all"))


# 2. CREATE A WORKING COPY

clean_df = df.copy()


# 3. CHECK MISSING VALUES

print("\nMISSING VALUE ANALYSIS")

print("\nMissing values in each column:")
print(clean_df.isnull().sum())

print("\nTotal missing values:")
print(clean_df.isnull().sum().sum())


# 4. REMOVE DUPLICATE ROWS

print("\nDUPLICATE ANALYSIS")

duplicates_before = clean_df.duplicated().sum()

print("Duplicate rows before removal:", duplicates_before)

clean_df = clean_df.drop_duplicates()

print("Duplicate rows after removal:", clean_df.duplicated().sum())

print("Shape after removing duplicates:", clean_df.shape)


# 5. CLEAN COLUMN NAMES AND TEXT VALUES

clean_df.columns = clean_df.columns.str.strip()

for column in clean_df.select_dtypes(include="object").columns:
    clean_df[column] = clean_df[column].str.strip()


# 6. REMOVE COMPLETELY EMPTY COLUMNS

print("\nEMPTY COLUMN ANALYSIS")

empty_columns = clean_df.columns[
    clean_df.isnull().all()
].tolist()

print("Completely empty columns:", empty_columns)

clean_df.drop(columns=empty_columns, inplace=True)


# 7. HANDLE MISSING VALUES

print("\nFILLING MISSING VALUES")

numerical_columns = clean_df.select_dtypes(
    include=np.number
).columns

categorical_columns = clean_df.select_dtypes(
    include=["object", "category"]
).columns


# Handle numerical missing values

for column in numerical_columns:

    if clean_df[column].isnull().sum() > 0:

        # Special handling for identifier and count columns

        if column in ["agent", "company", "children"]:

            fill_value = 0

            clean_df[column] = clean_df[column].fillna(fill_value)

            print(f"{column}: Missing values filled with 0")

        else:

            skewness = clean_df[column].skew()

            if abs(skewness) < 0.5:
                fill_value = clean_df[column].mean()
                method = "Mean"

            else:
                fill_value = clean_df[column].median()
                method = "Median"

            clean_df[column] = clean_df[column].fillna(fill_value)

            print(f"{column}: Missing values filled using {method}")


# Handle categorical missing values

for column in categorical_columns:

    if clean_df[column].isnull().sum() > 0:

        if column == "country":

            clean_df[column] = clean_df[column].fillna("Unknown")

            print(f"{column}: Missing values filled with Unknown")

        else:

            mode_value = clean_df[column].mode()

            if not mode_value.empty:

                clean_df[column] = clean_df[column].fillna(
                    mode_value[0]
                )

                print(f"{column}: Missing values filled using Mode")


# 8. CORRECT DATA TYPES

print("\nDATA TYPE CORRECTION")

clean_df["is_canceled"] = clean_df["is_canceled"].astype("int8")

count_columns = [
    "lead_time",
    "arrival_date_year",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "total_of_special_requests"
]

for column in count_columns:

    if column in clean_df.columns:
        clean_df[column] = clean_df[column].round().astype("int64")


# Convert selected categorical columns

categorical_columns = [
    "hotel",
    "meal",
    "country",
    "market_segment",
    "deposit_type"
]

for column in categorical_columns:

    if column in clean_df.columns:
        clean_df[column] = clean_df[column].astype("category")


# Set arrival month as an ordered category

months = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

if "arrival_date_month" in clean_df.columns:

    clean_df["arrival_date_month"] = pd.Categorical(
        clean_df["arrival_date_month"],
        categories=months,
        ordered=True
    )

print("\nUpdated data types:")
print(clean_df.dtypes)


# 9. REMOVE CONSTANT COLUMNS

print("\nCONSTANT COLUMN ANALYSIS")

constant_columns = [
    column for column in clean_df.columns
    if clean_df[column].nunique(dropna=False) <= 1
]

print("Constant columns:", constant_columns)

if constant_columns:

    clean_df.drop(columns=constant_columns, inplace=True)

    print("Removed constant columns:", constant_columns)

else:

    print("No constant columns found.")


# 10. DATA VALIDATION

print("\nDATA VALIDATION")

non_negative_columns = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "adr",
    "total_of_special_requests"
]

for column in non_negative_columns:

    if column in clean_df.columns:

        invalid_count = (clean_df[column] < 0).sum()

        print(f"Negative values in {column}: {invalid_count}")

# 11. FEATURE ENGINEERING

print("\nFEATURE ENGINEERING")

clean_df["total_nights"] = (
    clean_df["stays_in_weekend_nights"] +
    clean_df["stays_in_week_nights"]
)

clean_df["total_guests"] = (
    clean_df["adults"] +
    clean_df["children"]
)

clean_df["is_family"] = (
    clean_df["children"] > 0
).astype("int8")

print("\nNew features created:")
print("1. total_nights")
print("2. total_guests")
print("3. is_family")


# 12. FINAL MISSING VALUE CHECK

print("\nFINAL MISSING VALUE CHECK")

print(clean_df.isnull().sum())

print("\nTotal missing values remaining:")
print(clean_df.isnull().sum().sum())


# 13. FINAL DUPLICATE CHECK

print("\nFINAL DUPLICATE CHECK")

print("Duplicate rows remaining:", clean_df.duplicated().sum())


# 14. FINAL DATASET INFORMATION

print("\nCLEANED DATASET")

print("\nFirst 5 rows:")
print(clean_df.head())

print("\nLast 5 rows:")
print(clean_df.tail())

print("\nRandom 5 rows:")
print(clean_df.sample(min(5, len(clean_df)), random_state=42))

print("\nFinal shape:")
print(clean_df.shape)

print("\nFinal column names:")
print(clean_df.columns.tolist())

print("\nFinal dataset information:")
clean_df.info()

print("\nFinal statistical summary:")
print(clean_df.describe(include="all"))


# 15. SAVE CLEANED DATASET

clean_df.to_csv("Hotel_Booking_Cleaned.csv", index=False)

print("\nDATA CLEANING COMPLETED SUCCESSFULLY!")

print("Cleaned dataset saved as:")
print("Hotel_Booking_Cleaned.csv")

# DATA ANALYSIS USING PANDAS AND NUMPY

print("\n------------HOTEL BOOKING DATA ANALYSIS------------")


# GENERAL DATASET ANALYSIS

print("\nGENERAL DATASET ANALYSIS")

print("Total Bookings:", len(clean_df))
print("Total Columns:", clean_df.shape[1])
print("Total Hotels:", clean_df["hotel"].nunique())

# HOTEL-WISE BOOKING ANALYSIS

print("\nHOTEL-WISE BOOKING ANALYSIS")

hotel_analysis = clean_df["hotel"].value_counts()

print("\nBookings by Hotel:")
print(hotel_analysis)

print("\nMost Booked Hotel:")
print(hotel_analysis.idxmax())

# BOOKING CANCELLATION ANALYSIS

print("\nBOOKING CANCELLATION ANALYSIS")

cancellation_count = clean_df["is_canceled"].value_counts().sort_index()

print("\nCancellation Count:")
print(cancellation_count)

print("\nCancellation Percentage:")
print(clean_df["is_canceled"].value_counts(normalize=True) * 100)

print("\nTotal Cancelled Bookings:",
      (clean_df["is_canceled"] == 1).sum())

print("Total Non-Cancelled Bookings:",
      (clean_df["is_canceled"] == 0).sum())

print("Overall Cancellation Rate:",
      round(clean_df["is_canceled"].mean() * 100, 2), "%")

# MONTH-WISE BOOKING ANALYSIS

print("\nMONTH-WISE BOOKING ANALYSIS")

monthly_analysis = (
    clean_df["arrival_date_month"]
    .value_counts()
    .reindex(months, fill_value=0)
)

print("\nBookings in Each Month:")
print(monthly_analysis)

print("\nMonth with Maximum Bookings:")
print(monthly_analysis.idxmax())

print("\nMonth with Minimum Bookings:")
print(monthly_analysis.idxmin())


# COUNTRY-WISE BOOKING ANALYSIS

print("\nCOUNTRY-WISE BOOKING ANALYSIS")

country_analysis = clean_df["country"].value_counts()

print("\nTop 10 Countries:")
print(country_analysis.head(10))

print("\nCountry with Maximum Bookings:")
print(country_analysis.idxmax())

# LENGTH OF STAY ANALYSIS

print("\nLENGTH OF STAY ANALYSIS")

print("Average Stay Duration:",
      round(clean_df["total_nights"].mean(), 2), "nights")

print("Maximum Stay Duration:",
      clean_df["total_nights"].max(), "nights")

print("Minimum Stay Duration:",
      clean_df["total_nights"].min(), "nights")

print("\nStay Duration Distribution:")
print(clean_df["total_nights"].value_counts().sort_index().head(15))


# ADR (AVERAGE DAILY RATE) ANALYSIS

print("\nADR ANALYSIS")

print("Average ADR:", round(clean_df["adr"].mean(), 2))
print("Median ADR:", round(clean_df["adr"].median(), 2))
print("Minimum ADR:", clean_df["adr"].min())
print("Maximum ADR:", clean_df["adr"].max())
print("ADR Standard Deviation:", round(clean_df["adr"].std(), 2))

# HOTEL-WISE CANCELLATION COMPARISON

print("\nHOTEL-WISE CANCELLATION COMPARISON")

hotel_cancellation = pd.crosstab(
    clean_df["hotel"],
    clean_df["is_canceled"],
    margins=True
)

print(hotel_cancellation)

# MONTH-WISE CANCELLATION ANALYSIS

print("\nMONTH-WISE CANCELLATION ANALYSIS")

monthly_cancellation = pd.crosstab(
    clean_df["arrival_date_month"],
    clean_df["is_canceled"]
).reindex(months, fill_value=0)

monthly_cancellation.columns = [
    "Not Cancelled", "Cancelled"
][:len(monthly_cancellation.columns)]

print(monthly_cancellation)

# 18. FINAL STATISTICAL SUMMARY

print("\n18. FINAL STATISTICAL SUMMARY")

print(clean_df.describe(include="all"))


# FINAL ANALYSIS RESULTS

print("\n---------- FINAL ANALYSIS RESULTS ----------")

print("Total Bookings:", len(clean_df))

print("Most Booked Hotel:",
      clean_df["hotel"].value_counts().idxmax())

print("Most Popular Month:",
      monthly_analysis.idxmax())

print("Top Booking Country:",
      clean_df["country"].value_counts().idxmax())

print("Overall Cancellation Rate:",
      round(clean_df["is_canceled"].mean() * 100, 2), "%")

print("Average ADR:",
      round(clean_df["adr"].mean(), 2))

print("Average Stay:",
      round(clean_df["total_nights"].mean(), 2), "nights")

print("Family Bookings:",
      (clean_df["is_family"] == 1).sum())

print("\nDATA ANALYSIS COMPLETED SUCCESSFULLY!")
