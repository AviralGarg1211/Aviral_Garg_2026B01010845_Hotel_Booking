
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
