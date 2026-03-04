# **ECONOMY, INDUSTRY AND SUSTAINABILITY**
# **Spatial-Temporal Modeling of the Relationship between Individual Industries and the Economy using Panel Time Series Modeling**


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


<img width="377" height="726" alt="image" src="https://github.com/user-attachments/assets/faf3e92a-45bf-4fef-a5c0-d0de0f41ae2b" />


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

### Random intercepts for GDP and Brewery Count

```Python

model_desc = """
# Random intercepts
RI_GDP =~ 1*GDP
RI_brewery_count =~ 1*brewery_count

# Cross-lagged effects
GDP ~ a*brewery_count + b*GDP_lag + c*Household_Income
brewery_count ~ d*GDP + e*GDP_lag + f*Household_Income

# Covariances between residuals
GDP ~~ brewery_count
"""

```

### Spatial Cross-lagged Panel Model (SCLPM)

```Python

model_desc = """
log_GDP ~
          a*log_GDP_lag +
          b*log_income_lag +
          c*W_GDP_lag +
          d*W_brewery_count +
          e*brewery_count
"""

```

### Arellano-Bond Difference Generalized Method of Moments (GMM)

```Python

# Initializing Variables
# Dependent variable
y = gmm_data["log_GDP"]

# Endogenous regressor
endog = gmm_data["log_GDP_lag"]

# Exogenous variables (differences)
exog_vars = ["log_income", "brewery_count", "W_GDP_lag", "W_brewery_count"]
exog_vars = [v for v in exog_vars if v in gmm_data.columns]
exog = gmm_data[exog_vars]

# Aligning instrument to differenced dataframe
instruments = gmm_data["log_GDP_lag2"]

# Estimating GMM Model
model = IVGMM(
    dependent=y,
    exog=exog,
    endog=endog,
    instruments=instruments
)

```
