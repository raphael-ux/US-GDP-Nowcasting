import pandas as pd
from pathlib import Path
import config
import numpy as np 

class DataProcessing:
    '''
    do the preprocessing of our labels and features 
    '''
    

    def __init__(self):

        self.project_root = Path(__file__).resolve().parents[1]

        self.raw_gdp_path = (
            self.project_root
            / "data"
            / "raw"
            / "calendar_gdp.csv"
        )

        self.raw_features_path = (
            self.project_root
            / "data"
            / "raw"
            / "raw_features.pkl"
        )
        
        self.processed_features_path = (
            self.project_root
            / "data"
            / "processed"
            / "processed_features.csv"
        )
        
        self.alternative_news_data_path = (
            self.project_root
            /"data"
            /"raw"
            /"alternative"
            /"sf_fed_news_sentiment.csv"
        )
        
        self.final_df_path = (
            self.project_root
            / "data"
            / "processed"
            / "final_df.csv"
        )

        self.calendar = pd.read_csv(
            self.raw_gdp_path,
            parse_dates=[
                "quarter_start",
                "quarter_end",
                "release_date"
            ]
        )
        
        self.alternative_news_data = pd.read_csv(
            self.alternative_news_data_path,
            parse_dates=["observation_date"]
        )
        self.raw_features = pd.read_pickle(
            self.raw_features_path
        )
        
        self.preprocessing_config = config.PREPROCESSING_CONFIG
        
        self.processed_features = None
        self.preprocessed_labels = None
        self.features_df = None
        
    
        
    def _process_series(
        self,
        series_df,
        series_id,
        config
    ):

        feature_name = config["feature_name"]
        transform = config["transform"]
        windows = config["windows"]

        values = (
            series_df["value"]
            .dropna()
            .reset_index(drop=True)
        )

        features = {}

        if len(values) == 0:
            return features

        if transform == "log_growth":

            for window in windows:

                if len(values) > window:

                    growth = (
                        100
                        * (
                            np.log(values.iloc[-1])
                            - np.log(values.iloc[-1 - window])
                        )
                    )

                    features[
                        f"{feature_name}_growth_{window}"
                    ] = growth

        elif transform == "difference":

            for window in windows:

                if len(values) > window:

                    difference = (
                        values.iloc[-1]
                        - values.iloc[-1 - window]
                    )

                    features[
                        f"{feature_name}_diff_{window}"
                    ] = difference

        elif transform == "level_mean":

            for window in windows:

                if len(values) >= window:

                    mean_value = (
                        values
                        .iloc[-window:]
                        .mean()
                    )

                    features[
                        f"{feature_name}_mean_{window}"
                    ] = mean_value

        return features
    
    
    def _build_news_features(
        self,
        publication_lag_days=7
    ):

        news = (
            self.alternative_news_data
            .copy()
            .sort_values("observation_date")
            .reset_index(drop=True)
        )

        # Use date temporarily as index for time-based rolling
        news = news.set_index(
            "observation_date"
        )

        # Latest sentiment
        news[
            "news_sentiment_latest"
        ] = news[
            "news_sentiment"
        ]

        # 30-day rolling mean
        news[
            "news_sentiment_mean_30d"
        ] = (
            news["news_sentiment"]
            .rolling(
                "30D",
                min_periods=1
            )
            .mean()
        )

        # 90-day rolling mean
        news[
            "news_sentiment_mean_90d"
        ] = (
            news["news_sentiment"]
            .rolling(
                "90D",
                min_periods=1
            )
            .mean()
        )

        # Momentum
        news[
            "news_sentiment_momentum"
        ] = (
            news["news_sentiment_mean_30d"]
            - news["news_sentiment_mean_90d"]
        )

        # Restore observation_date as a normal column
        news = news.reset_index()

        # Conservative publication lag
        news[
            "availability_date"
        ] = (
            news["observation_date"]
            + pd.Timedelta(
                days=publication_lag_days
            )
        )

        return news


    def _align_news_to_cutoffs(
        self,
        news_features,
        cutoff_dates
    ):

        cutoffs = pd.DataFrame({
            "cutoff_date":
                pd.to_datetime(
                    cutoff_dates
                )
        })

        cutoffs = (
            cutoffs
            .sort_values("cutoff_date")
            .reset_index(drop=True)
        )

        news_features = (
            news_features
            .sort_values(
                "availability_date"
            )
            .reset_index(drop=True)
        )

        news_at_cutoff = (
            pd.merge_asof(
                cutoffs,
                news_features,
                left_on="cutoff_date",
                right_on="availability_date",
                direction="backward"
            )
        )

        return news_at_cutoff
        
    def build_features(
        self,
        include_news=True,
        news_lag_days=7
    ):

        # =============================================
        # 1. FRED FEATURES
        # =============================================

        rows = []

        for cutoff_date, snapshot in (
            self.raw_features.items()
        ):

            row = {
                "cutoff_date":
                    pd.Timestamp(
                        cutoff_date
                    )
            }

            for (
                series_id,
                series_config
            ) in (
                self.preprocessing_config.items()
            ):

                series_df = snapshot[
                    snapshot["series_id"]
                    == series_id
                ]

                features = (
                    self._process_series(
                        series_df=series_df,
                        series_id=series_id,
                        config=series_config
                    )
                )

                row.update(
                    features
                )

            rows.append(
                row
            )

        features_df = pd.DataFrame(
            rows
        )

        features_df = (
            features_df
            .sort_values("cutoff_date")
            .reset_index(drop=True)
        )

        # =============================================
        # 2. NEWS FEATURES
        # =============================================

        if include_news:

            news_daily = (
                self._build_news_features(
                    publication_lag_days=
                        news_lag_days
                )
            )

            news_cutoff = (
                self._align_news_to_cutoffs(
                    news_features=
                        news_daily,
                    cutoff_dates=
                        features_df[
                            "cutoff_date"
                        ]
                )
            )

            # -----------------------------------------
            # Keep only model features
            # -----------------------------------------

            news_columns = [
                "cutoff_date",
                "news_sentiment_latest",
                "news_sentiment_mean_30d",
                "news_sentiment_mean_90d",
                "news_sentiment_momentum"
            ]

            news_cutoff = (
                news_cutoff[
                    news_columns
                ]
            )

            features_df = (
                features_df
                .merge(
                    news_cutoff,
                    on="cutoff_date",
                    how="left"
                )
            )

        # =============================================
        # 3. SAVE
        # =============================================

        self.processed_features = (
            features_df
        )

        features_df.to_csv(
            self.processed_features_path,
            index=False
        )
        
        self.features_df = features_df

        return features_df
        
        
        
    def features_cleaning(
        self,
        missing_threshold=0.10
    ):

        df = self.features_df.copy()

        # ---------------------------------------
        # Compute missing ratio per feature
        # ---------------------------------------

        missing_ratio = (
            df
            .isna()
            .mean()
        )

        # ---------------------------------------
        # Features to remove
        # ---------------------------------------

        features_to_drop = (
            missing_ratio[
                missing_ratio >= missing_threshold
            ]
            .index
            .tolist()
        )

        # ---------------------------------------
        # Remove features
        # ---------------------------------------
        
        print(f'features to dropp ; {features_to_drop}')

        cleaned_df = df.drop(
            columns=features_to_drop
        )

        # ---------------------------------------
        # Save results
        # ---------------------------------------

        self.cleaned_features = cleaned_df

        self.features_dropped_missing = (
            features_to_drop
        )

        self.missing_ratio = (
            missing_ratio
        )

        print(
            f"Original features: {df.shape[1]}"
        )

        print(
            f"Dropped features: "
            f"{len(features_to_drop)}"
        )

        print(
            f"Remaining features: "
            f"{cleaned_df.shape[1]}"
        )

        if features_to_drop:

            print(
                "\nDropped because of missingness:"
            )

            for feature in features_to_drop:

                print(
                    f"{feature}: "
                    f"{missing_ratio[feature] * 100:.2f}%"
                )
        
        self.cleaned_features.to_csv(self.processed_features_path,index = True)

        return self.cleaned_features
        
    def build_labels(
        self,
        label_type="level"
    ):
        """
        Build GDP target variables.

        Parameters
        ----------
        label_type : str

            "level"
                GDP_t

            "difference"
                GDP_t - GDP_{t-1}

            "pct_change"
                100 * (GDP_t / GDP_{t-1} - 1)

            "log_growth"
                100 * [log(GDP_t) - log(GDP_{t-1})]

            "annualized_log_growth"
                400 * [log(GDP_t) - log(GDP_{t-1})]

        Returns
        -------
        pd.DataFrame
        """

        labels_df = (
            self.calendar[
                [
                    "quarter_end",
                    "initial_value",
                    "initial_growth",
                    "cutoff_date"
                ]
            ]
            .copy()
            .rename(
                columns={
                    "initial_value": "gdp_initial"
                }
            )
            .sort_values("cutoff_date")
            .reset_index(drop=True)
        )

        labels_df["last_quarter_gdp"] = (
            labels_df["gdp_initial"]
            .shift(1)
        )

        if label_type == "level":

            labels_df["target"] = (
                labels_df["gdp_initial"]
            )

        elif label_type == "difference":

            labels_df["target"] = (
                labels_df["gdp_initial"]
                - labels_df["last_quarter_gdp"]
            )

        elif label_type == "pct_change":

            labels_df["target"] = (
                100
                * (
                    labels_df["gdp_initial"]
                    / labels_df["last_quarter_gdp"]
                    - 1
                )
            )

        elif label_type == "log_growth":

            labels_df["target"] = (
                100
                * (
                    np.log(
                        labels_df["gdp_initial"]
                    )
                    - np.log(
                        labels_df["last_quarter_gdp"]
                    )
                )
            )

        elif label_type == "annualized_log_growth":

            labels_df["target"] = (
                400
                * (
                    np.log(
                        labels_df["gdp_initial"]
                    )
                    - np.log(
                        labels_df["last_quarter_gdp"]
                    )
                )
            )
            
        elif label_type == "annualized_growth":

            labels_df["target"] = (
                labels_df["initial_growth"]
            )
        

        else:

            raise ValueError(
                "label_type must be one of: "
                "'level', "
                "'difference', "
                "'pct_change', "
                "'log_growth', "
                "'annualized_log_growth', "
                "'annualized_growth'"
            )

        self.preprocessed_labels = labels_df

        return self.preprocessed_labels
    
    def create_final_df(self):

        if self.cleaned_features is None:

            features_df = pd.read_csv(
                self.processed_features_path,
                parse_dates=["cutoff_date"]
            )

        else:
            
            features_df = (
                self.cleaned_features
                .reset_index(drop=True)
            )

        if self.preprocessed_labels is None:

            raise ValueError(
                "Labels have not been built. "
                "Call build_labels() first."
            )

        features_df["cutoff_date"] = pd.to_datetime(features_df["cutoff_date"])
        self.preprocessed_labels["cutoff_date"] = pd.to_datetime(self.preprocessed_labels["cutoff_date"]) 
        
        final_df = features_df.merge(
            self.preprocessed_labels,
            on="cutoff_date",
            how="inner"
        )

        final_df = (
            final_df
            .sort_values("cutoff_date")
            .reset_index(drop=True)
        )

        final_df.to_csv(
            self.final_df_path,
            index=False
        )

        self.final_df = final_df

        return self.final_df
        
    
    
        
        
    