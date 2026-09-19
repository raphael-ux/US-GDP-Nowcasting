FRED_FEATURES = {

    # --- Consumption ---
    "PCEC96": "real_pce", 
    "RSAFS": "retail_sales",
    "TOTALSA": "vehicle_sales",
    "DSPIC96": "real_disposable_income",

    # --- Residential investment ---
    "HOUST": "housing_starts",
    "PERMIT": "building_permits",
    "TLRESCONS": "residential_construction",

    # --- Business investment ---
    "ANXAVS": "capital_goods_shipments",
    "TLNRESCONS": "nonresidential_construction",
    "AMTMNO": "manufacturing_new_orders",

    # --- Inventories ---
    "BUSINV": "business_inventories",

    #--- Net exports ---
    "BOPTEXP": "exports",
    "BOPTIMP": "imports",

    # --- Government ---
    "TLPBLCONS": "public_construction",

    # --- Production ---
    "INDPRO": "industrial_production",

    # --- Labour ---
    "PAYEMS": "nonfarm_payrolls",
    "ICSA": "initial_claims",

    # --- Prices ---
    "PCEPI": "pce_price_index",
    "CPIAUCSL": "cpi",

    # --- Broad activity ---
    "CFNAI": "national_activity_index",

    # --- Financial conditions ---
    "NFCI": "financial_conditions",
}



#from build_preprocessing_config by DataCollection class

PREPROCESSING_CONFIG = {

    "PCEC96": {
        "feature_name": "real_pce",
        "frequency": "M",
        "units": "Billions of Chained 2017 Dollars",
        "seasonal_adjustment": "SAAR",

        "transform": "log_growth",
        "windows": [1, 3],
    },

    "RSAFS": {
        "feature_name": "retail_sales",
        "frequency": "M",
        "units": "Millions of Dollars",
        "seasonal_adjustment": "SA",

        "transform": "log_growth",
        "windows": [1, 3],
    },

    "TOTALSA": {
        "feature_name": "vehicle_sales",
        "frequency": "M",
        "units": "Millions of Units",
        "seasonal_adjustment": "SAAR",

        "transform": "log_growth",
        "windows": [1, 3],
    },

    "DSPIC96": {
        "feature_name": "real_disposable_income",
        "frequency": "M",
        "units": "Billions of Chained 2017 Dollars",
        "seasonal_adjustment": "SAAR",

        "transform": "log_growth",
        "windows": [1, 3],
    },

    "HOUST": {
        "feature_name": "housing_starts",
        "frequency": "M",
        "units": "Thousands of Units",
        "seasonal_adjustment": "SAAR",

        "transform": "log_growth",
        "windows": [1, 3],
    },

    "PERMIT": {
        "feature_name": "building_permits",
        "frequency": "M",
        "units": "Thousands of Units",
        "seasonal_adjustment": "SAAR",

        "transform": "log_growth",
        "windows": [1, 3],
    },

    "TLRESCONS": {
        "feature_name": "residential_construction",
        "frequency": "M",
        "units": "Millions of Dollars",
        "seasonal_adjustment": "SAAR",

        "transform": "log_growth",
        "windows": [1, 3],
    },

    "ANXAVS": {
        "feature_name": "capital_goods_shipments",
        "frequency": "M",
        "units": "Millions of Dollars",
        "seasonal_adjustment": "SA",

        "transform": "log_growth",
        "windows": [1, 3],
    },

    "TLNRESCONS": {
        "feature_name": "nonresidential_construction",
        "frequency": "M",
        "units": "Millions of Dollars",
        "seasonal_adjustment": "SAAR",

        "transform": "log_growth",
        "windows": [1, 3],
    },

    "AMTMNO": {
        "feature_name": "manufacturing_new_orders",
        "frequency": "M",
        "units": "Millions of Dollars",
        "seasonal_adjustment": "SA",

        "transform": "log_growth",
        "windows": [1, 3],
    },

    "BUSINV": {
        "feature_name": "business_inventories",
        "frequency": "M",
        "units": "Millions of Dollars",
        "seasonal_adjustment": "SA",

        "transform": "difference",
        "windows": [1, 3],
    },

    "BOPTEXP": {
        "feature_name": "exports",
        "frequency": "M",
        "units": "Millions of Dollars",
        "seasonal_adjustment": "SA",

        "transform": "log_growth",
        "windows": [1, 3],
    },

    "BOPTIMP": {
        "feature_name": "imports",
        "frequency": "M",
        "units": "Millions of Dollars",
        "seasonal_adjustment": "SA",

        "transform": "log_growth",
        "windows": [1, 3],
    },

    "TLPBLCONS": {
        "feature_name": "public_construction",
        "frequency": "M",
        "units": "Millions of Dollars",
        "seasonal_adjustment": "SAAR",

        "transform": "log_growth",
        "windows": [1, 3],
    },

    "INDPRO": {
        "feature_name": "industrial_production",
        "frequency": "M",
        "units": "Index 2017=100",
        "seasonal_adjustment": "SA",

        "transform": "log_growth",
        "windows": [1, 3],
    },

    "PAYEMS": {
        "feature_name": "nonfarm_payrolls",
        "frequency": "M",
        "units": "Thousands of Persons",
        "seasonal_adjustment": "SA",

        "transform": "difference",
        "windows": [1, 3],
    },

    "ICSA": {
        "feature_name": "initial_claims",
        "frequency": "W",
        "units": "Number",
        "seasonal_adjustment": "SA",

        "transform": "level_mean",
        "windows": [1, 4, 13],
    },

    "PCEPI": {
        "feature_name": "pce_price_index",
        "frequency": "M",
        "units": "Index 2017=100",
        "seasonal_adjustment": "SA",

        "transform": "log_growth",
        "windows": [1, 3],
    },

    "CPIAUCSL": {
        "feature_name": "cpi",
        "frequency": "M",
        "units": "Index 1982-1984=100",
        "seasonal_adjustment": "SA",

        "transform": "log_growth",
        "windows": [1, 3],
    },

    "CFNAI": {
        "feature_name": "national_activity_index",
        "frequency": "M",
        "units": "Index",
        "seasonal_adjustment": "NSA",

        "transform": "level_mean",
        "windows": [1, 3],
    },

    "NFCI": {
        "feature_name": "financial_conditions",
        "frequency": "W",
        "units": "Index",
        "seasonal_adjustment": "NSA",

        "transform": "level_mean",
        "windows": [1, 4, 13],
    },
}

target_col = "gdp_initial"