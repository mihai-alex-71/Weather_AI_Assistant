import numpy as np
import pandas as pd
from tqdm import tqdm

source_csv = "weather_romania_38_cities_2021_2025.csv"
output_csv = "training_6h_dataset.csv"

Past_hours = 24
future_Horizon = [1, 2, 3, 4, 5, 6]  # times that the model is going to predict

TIME_COL = "time"
CITY_COL = "city_id"
WMO_COL = "weather_code"

BASE_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
    "weather_code",
    "latitude",
    "longitude",
    "elevation",
]


def map_wmo_to_condition(wmo: int) -> int:
    if wmo in [0, 1, 2, 3]:
        return 0
    if 40 <= wmo <= 49:
        return 1  # fog
    if 51 <= wmo <= 67:
        return 2  # rain
    if 80 <= wmo <= 82:
        return 2  # rain
    if (71 <= wmo <= 77) or wmo in [85, 86]:
        return 3  # snow / ice
    if 95 <= wmo <= 99:
        return 4  # thunderstorm
    return 0


print("Loading CSV...")
df = pd.read_csv(source_csv)
print("loaded csv file")


df[TIME_COL] = pd.to_datetime(df[TIME_COL])
df = df.sort_values([CITY_COL, TIME_COL]).reset_index(drop=True)
# sort first by city then time - already have but garanteed.


first_chunk = True

for city_id, city_df in df.groupby(CITY_COL):
    city_df = city_df.sort_values(TIME_COL).reset_index(drop=True)

    n = len(city_df)

    min_idx = Past_hours  # enough past data
    max_idx = n - max(future_Horizon) - 1  # enough future data

    if max_idx <= min_idx:
        print(f"  Not enough data for city {city_id}, skipping")
        continue

    rows = []

    # sampling
    for i in tqdm(range(min_idx, max_idx+1), desc=f"city {city_id}", leave=False):
        sample = {}
        # time stamp

        for lag in range(1, Past_hours+1):
            row_lag = city_df.iloc[i - lag]
            for feature in BASE_FEATURES:
                sample[f"{feature}_lag_{lag}h"] = row_lag[feature]

        current_row = city_df.iloc[i]

        sample["city_id"] = city_id
        sample["time"] = current_row[TIME_COL]
        sample["hour"] = current_row[TIME_COL].hour
        sample["dayofweek"] = current_row[TIME_COL].dayofweek
        sample["month"] = current_row[TIME_COL].month

        # y targets
        for h in future_Horizon:
            future_row = city_df.iloc[i+h]

            sample[f"target_wind_speed_h{h}"] = future_row["wind_speed_10m"]
            sample[f"target_temperature_h{h}"] = future_row["temperature_2m"]
            sample[f"target_precipitation_h{h}"] = future_row["precipitation"]

            wmo_val = future_row[WMO_COL]
            if pd.isna(wmo_val):  # had some Wmo code missing 52 rows
                sample = None
                break

            wmo_code = int(wmo_val)
            sample[f"target_wmo_class_h{h}"] = map_wmo_to_condition(wmo_code)
        if sample is not None:
            rows.append(sample)

    city_training_df = pd.DataFrame(rows)

    meta_cols = ["city_id", "time", "hour", "dayofweek", "month"]
    target_cols = sorted(
        [c for c in city_training_df.columns if c.startswith("target_")])
    exclude = set(target_cols + ["city_id", "time"])
    feature_cols = [c for c in city_training_df.columns if c not in exclude]

    city_training_df = city_training_df[meta_cols + feature_cols + target_cols]

    if first_chunk:
        city_training_df.to_csv(output_csv, index=False, mode="w")
        first_chunk = False
    else:
        city_training_df.to_csv(output_csv, index=False,
                                mode="a", header=False)

    # free memory
    del rows
    del city_training_df

print("done!")
