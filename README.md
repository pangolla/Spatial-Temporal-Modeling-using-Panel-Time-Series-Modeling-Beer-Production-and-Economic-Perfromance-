# **Spatial-Temporal Modeling of the Relationship between the Performance of the Beer industry and the Economy across the United States using Panel Time Series Modeling**

This project combines spatial and temporal modeling in python using the panel time series modeling for the case of the beer industry to evaluate:
- Bi-directional relationship for the mutual effects between beer industry performance and economic performance.
- Spatial effects on beer industry performance
- Spatial effects on economic performance
- Temporal effects on beer industry performance
- Temporal effects on economic performance

## Models Implemented

The following three models were implemented by the code:
- Random Intercept Cross-lagged Panel Model (RI-CLPM)
- Spatial Cross-lagged Panel Model (SCLPM)
- Arellano-Bond Difference Generalized Method of Moments (GMM)

## Prerequisites

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.affinity import scale, translate
from semopy import Model
from libpysal.weights import Queen
from libpysal.weights import lag_spatial
from linearmodels.iv import IVGMM

## Implementation 

### Random Intercept Cross-lagged Panel Model (RI-CLPM)

model_desc = """
Random intercepts
RI_GDP =~ 1*GDP
RI_brewery_count =~ 1*brewery_count

Cross-lagged effects
GDP ~ a*brewery_count + b*GDP_lag + c*Household_Income
brewery_count ~ d*GDP + e*GDP_lag + f*Household_Income

Covariances between residuals
GDP ~~ brewery_count
"""

### Spatial Cross-lagged Panel Model (SCLPM)

model_desc = """
log_GDP ~ a*log_GDP_lag + 
          b*log_income_lag + 
          c*W_GDP_lag + 
          d*W_brewery_count + 
          e*brewery_count
"""

### Arellano-Bond Difference Generalized Method of Moments (GMM)

Set multi-index (required for panel GMM) and sorting data
merged_data_mi = merged_data.set_index(["State", "Year"])
merged_data_mi = merged_data_mi.sort_index()

Getting lag 2 for dependent variable
merged_data_mi["log_GDP_lag2"] = (
    merged_data_mi.groupby(level=0)["log_GDP"].shift(2)
)

Getting First Difference
data_diff = merged_data_mi.groupby(level=0).diff()

Combining differenced variables with instrument in one dataframe
gmm_data = pd.concat([
    data_diff["log_GDP"],
    data_diff["log_GDP_lag"],
    data_diff[[
        "log_income",
        "brewery_count",
        "W_GDP_lag",
        "W_income_lag"
    ]],
    merged_data_mi["log_GDP_lag2"]
], axis=1).dropna()

Removing columns with no variation (fixes rank issue)
gmm_data = gmm_data.loc[:, gmm_data.std() > 0]

Initializing Variables
Dependent variable
y = gmm_data["log_GDP"]

Endogenous regressor
endog = gmm_data["log_GDP_lag"]

Exogenous variables (differences)
exog_vars = ["log_income", "brewery_count", "W_GDP_lag", "W_income_lag"]
exog_vars = [v for v in exog_vars if v in gmm_data.columns]
exog = gmm_data[exog_vars]

Aligning instrument to differenced dataframe
instruments = gmm_data["log_GDP_lag2"]

Estimating GMM Model
model = IVGMM(
    dependent=y,
    exog=exog,
    endog=endog,
    instruments=instruments
)

