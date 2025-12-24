# Weather Assistant – Short-Term Forecast Model

**Author:** Shafiei Armin - Turcu Mihai Alexandru - Scrob Sebastian  

## Description

This project builds a short-term weather prediction dataset for Romania’s main cities. It uses a public historical weather API (hourly) to download past weather for the last 4 years (temperature, humidity, pressure, wind, precipitation, cloud cover, weather code, elevation). The data is then merged into a single “mega” CSV file that will later be used to train machine learning models for 1–6 hour ahead forecasts per city.

## Setup

1. Clone this repository or download the project files.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Step 1 – Download city weather data

Run:

```bash
python retrieve_data.py
```
**make sure you have around 2GB storage at your directory**
- This script calls the historical weather API for each configured Romanian city 
- It saves one CSV file per city .

## Step 2 – Merge into a single dataset

After the per-city downloads are complete, run:

```bash
python merge_data.py
```
- this script will simply adds all csv files on top of one another and make the mega data set which later we can use for training the model.

## Step 3 – Build supervised training dataset

After the merged CSV is created (e.g. `weather_romania_38_cities_2021_2025.csv`), run:

```bash
python build_trainingset.py
```


This script:

- Sorts data by `city_id` and `time`, then builds **sliding windows** per city.
- Uses the **past 24 hours** of all core features as inputs:
  - all features plus,
- `hour`, `dayofweek`, `month`.
- Creates **targets** for horizons 1–6 hours ahead:
  - `target_wind_speed_h1..h6`
  - `target_temperature_h1..h6`
  - `target_precipitation_h1..h6`
  - `target_wmo_class_h1..h6` (categorical weather condition derived from WMO code).
- Saves the final supervised dataset as `training_6h_dataset.csv`.






