```python
import pandas as pd

from xclopz.XCLOpz import XCLOpz

df = pd.read_csv("CanTho_DischargeTimeseries_2000-2016.csv")

xcl = XCLOpz()

xcl.prepare_data(
    df, "Discharge", ["Timestamp", "DY", "QV2M", "WS2M", "RH2M"],
    copy_df=True
)

xcl.train_catboost()
xcl.train_lightgbm()
xcl.train_xgboost()

xcl.plot_feature_importance()
xcl.plot_predictions()

xcl.save_models()
```

```python
import joblib
import pandas as pd

print(joblib.load("models/results_20250119_215048.joblib"))

df = pd.DataFrame(
    {
        "PRECTOTCORR": [0, 0, 0, 0, 0, 0, 0],
        "T2M_min": [22.4, 22.9, 23.1, 22.2, 21.5, 21.7, 22.6],
        "T2M_max": [28.9, 31.5, 31, 31.2, 31.3, 32.4, 32.9],
        "Evaporation": [3.37, 4.11, 3.83, 4.37, 4.4, 4.53, 4.48],
        "MO": [1, 1, 1, 1, 1, 1, 1],
    }
)

df["T2M"] = (df["T2M_min"] + df["T2M_max"]) / 2

df = df[["PRECTOTCORR", "T2M", "Evaporation", "MO"]]

print(joblib.load("models/catboost_20250119_215048.joblib").predict(
    df
))

print(joblib.load("models/lightgbm_20250119_215048.joblib").predict(
    df
))

df = df[['MO', 'PRECTOTCORR', 'T2M', 'Evaporation']]

print(joblib.load("models/xgboost_20250119_215048.joblib").predict(
    df
))
```