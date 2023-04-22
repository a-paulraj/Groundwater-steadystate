# A Deep Learning Emulator for a Groundwater Model: Mapping Steady-State Water Table Depth 

A deep learning emulator is developed to predict steady-state WTD over the contiguous US (CONUS), with a focus on exploring the relationships between WTD, hydraulic conductivity (K), and precipitation minus evapotranspiration (PME). 

A CNN architecture is trained on thousands of 32km*32km simulated data from a physically-based hydrologic model ParFlow. Sensitivity analyses are conducted to evaluate model robustness. Uncertainty distributions for WTD, K, and PME are developed by injecting gaussian noise into the emulator; this enables an assessment of importance among various hydrological quantities in steady state WTD modeling.

This work was conducted as part of a research internship with the HydroGEN team at Princeton University.
