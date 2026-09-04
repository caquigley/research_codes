#%%
import os
import tempfile
import requests
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer
'''
This requires a different conda environment to run
conda activate nws_rain
cd repos/array_aggregator
python datamine_pull_rain.py

saves local file
nws_daily_precipitation.csv
'''

# ============================================================
# USER SETTINGS
# ============================================================

start_date = "2025-09-08"
end_date = "2025-11-15" #'2025-11-15'

# Two representative locations in your array
# Replace these with your actual coordinates.
points = {
    "HM": {
        "lat": 59.6155153,
        "lon": -151.1426301,
    },
    "KD": {
        "lat": 57.4411191,
        "lon": -152.3593147,
    },
}


# ============================================================
# NWS HRAP PROJECTION
# ============================================================

hrap_crs = CRS.from_proj4(
    "+proj=stere "
    "+lat_0=90 "
    "+lat_ts=60 "
    "+lon_0=-105 "
    "+x_0=0 "
    "+y_0=0 "
    "+a=6367470 "
    "+b=6367470 "
    "+units=m "
    "+no_defs"
)

transformer = Transformer.from_crs(
    "EPSG:4326",
    hrap_crs,
    always_xy=True,
)


# ============================================================
# CONVERT STATION LOCATIONS TO HRAP
# ============================================================

for name, point in points.items():

    x, y = transformer.transform(
        point["lon"],
        point["lat"],
    )

    point["x_hrap"] = x
    point["y_hrap"] = y

    print(
        f"{name}: "
        f"lon={point['lon']}, "
        f"lat={point['lat']} -> "
        f"x={x:.0f}, y={y:.0f}"
    )


# ============================================================
# DATE RANGE
# ============================================================

dates = pd.date_range(
    start=start_date,
    end=end_date,
    freq="D",
)


# ============================================================
# STORAGE FOR RESULTS
# ============================================================

results = []


# ============================================================
# LOOP THROUGH DAYS
# ============================================================

for date in dates:

    date_string = date.strftime("%Y%m%d")

    url = (
        "https://water.noaa.gov/resources/downloads/precip/stageIV/"
        f"{date:%Y/%m/%d}/"
        f"nws_precip_1day_{date_string}_ak.nc"
    )

    print()
    print(f"Processing {date_string}...")
    print(url)

    tmp_file = None

    try:

        # ----------------------------------------------------
        # Download file
        # ----------------------------------------------------

        response = requests.get(
            url,
            timeout=60,
        )

        if response.status_code != 200:

            print(
                f"  File unavailable "
                f"(HTTP {response.status_code})"
            )

            continue

        # ----------------------------------------------------
        # Create temporary file
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".nc",
            delete=False,
        ) as tmp:

            tmp.write(response.content)
            tmp_file = tmp.name

        # ----------------------------------------------------
        # Open NetCDF explicitly with netCDF4
        # ----------------------------------------------------

        ds = xr.open_dataset(
            tmp_file,
            engine="netcdf4",
        )

        # ----------------------------------------------------
        # Print grid information for the first file
        # ----------------------------------------------------

        if len(results) == 0:

            print()
            print("NWS grid information:")
            print(
                f"  x range: "
                f"{ds.x.min().item():.0f} -> "
                f"{ds.x.max().item():.0f}"
            )
            print(
                f"  y range: "
                f"{ds.y.min().item():.0f} -> "
                f"{ds.y.max().item():.0f}"
            )

            print()
            print("Nearest NWS grid cells:")

        # ----------------------------------------------------
        # Extract precipitation
        # ----------------------------------------------------

        daily_result = {
            "date": date,
        }

        for name, point in points.items():

            x = point["x_hrap"]
            y = point["y_hrap"]

            rain = ds["observation"].sel(
                x=x,
                y=y,
                method="nearest",
            )

            # Actual grid coordinates selected
            grid_x = rain.x.item()
            grid_y = rain.y.item()

            value = rain.item()

            daily_result[f"{name}_precip_inches"] = value

            # Print grid location
            if len(results) == 0:

                print(
                    f"  {name}: "
                    f"requested ({x:.0f}, {y:.0f}) -> "
                    f"grid ({grid_x:.0f}, {grid_y:.0f})"
                )

            print(
                f"  {name}: {value:.2f} inches"
            )

        results.append(daily_result)

        ds.close()

    except Exception as e:

        print(
            f"  ERROR processing "
            f"{date_string}: {e}"
        )

    finally:

        # ----------------------------------------------------
        # Delete temporary file
        # ----------------------------------------------------

        if tmp_file is not None:

            if os.path.exists(tmp_file):
                os.remove(tmp_file)


# ============================================================
# CREATE DATAFRAME
# ============================================================

rainfall = pd.DataFrame(results)
rainfall['HM_precip_cm'] = rainfall['HM_precip_inches']*2.54
rainfall['KD_precip_cm'] = rainfall['KD_precip_inches']*2.54
print()
print("=" * 60)
print("RESULTS")
print("=" * 60)

print(rainfall)


# ============================================================
# SAVE RESULTS
# ============================================================

rainfall.to_csv(
    "nws_daily_precipitation.csv",
    index=False,
)

print()
print(
    "Saved to nws_daily_precipitation.csv"
)