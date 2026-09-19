from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DataAnalysis:

    def __init__(self):

        # --------------------------------------------------
        # Project paths
        # --------------------------------------------------

        self.project_root = (
            Path(__file__)
            .resolve()
            .parents[1]
        )

        self.raw_labels_path = (
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

        self.final_df_path = (
            self.project_root
            / "data"
            / "processed"
            / "final_df.csv"
        )

        # --------------------------------------------------
        # Load datasets
        # --------------------------------------------------

        self.raw_labels = self._load_csv(
            self.raw_labels_path
        )

        self.raw_features = self._load_pickle(
            self.raw_features_path
        )

        self.processed_features = self._load_csv(
            self.processed_features_path
        )

        self.final_df = self._load_csv(
            self.final_df_path
        )

    # ==================================================
    # LOAD DATA
    # ==================================================

    @staticmethod
    def _load_csv(path):

        if not path.exists():
            print(
                f"Warning: file not found: {path}"
            )
            return None

        return pd.read_csv(path)

    @staticmethod
    def _load_pickle(path):

        if not path.exists():
            print(
                f"Warning: file not found: {path}"
            )
            return None

        return pd.read_pickle(path)

    # ==================================================
    # GENERAL OVERVIEW
    # ==================================================

    def overview(self):
        """
        General overview of available datasets.
        """

        rows = []

        datasets = {
            "raw_labels": self.raw_labels,
            "processed_features": self.processed_features,
            "final_df": self.final_df
        }

        for name, df in datasets.items():

            if df is None:
                continue

            rows.append({
                "dataset": name,
                "n_rows": df.shape[0],
                "n_columns": df.shape[1],
                "total_nan": df.isna().sum().sum(),
                "nan_pct": (
                    100
                    * df.isna().sum().sum()
                    / df.size
                )
            })

        # Raw features is a dictionary of snapshots
        if self.raw_features is not None:

            rows.append({
                "dataset": "raw_features",
                "n_rows": len(self.raw_features),
                "n_columns": np.nan,
                "total_nan": np.nan,
                "nan_pct": np.nan
            })

        return pd.DataFrame(rows)

    # ==================================================
    # LATEST RAW SNAPSHOT
    # ==================================================

    def latest_raw_snapshot(self):
        """
        Return the most recent point-in-time raw feature snapshot.
        """

        if self.raw_features is None:
            raise ValueError(
                "raw_features.pkl is not available."
            )

        latest_cutoff = max(
            self.raw_features.keys()
        )

        snapshot = (
            self.raw_features[
                latest_cutoff
            ]
            .copy()
        )

        print(
            f"Latest cutoff: "
            f"{pd.Timestamp(latest_cutoff).date()}"
        )

        print(
            f"Number of raw observations: "
            f"{len(snapshot)}"
        )

        print(
            f"Number of series: "
            f"{snapshot['series_id'].nunique()}"
        )

        return snapshot

    # ==================================================
    # RAW SNAPSHOT SUMMARY
    # ==================================================

    def latest_snapshot_summary(self):
        """
        Summary by series for the latest cutoff date.
        """

        snapshot = (
            self.latest_raw_snapshot()
        )

        summary = (
            snapshot
            .groupby(
                [
                    "series_id",
                    "feature_name"
                ]
            )
            .agg(
                n_obs=(
                    "value",
                    "size"
                ),
                n_missing=(
                    "value",
                    lambda x: x.isna().sum()
                ),
                mean=(
                    "value",
                    "mean"
                ),
                std=(
                    "value",
                    "std"
                ),
                min=(
                    "value",
                    "min"
                ),
                max=(
                    "value",
                    "max"
                ),
                first_obs=(
                    "observation_date",
                    "min"
                ),
                last_obs=(
                    "observation_date",
                    "max"
                )
            )
            .reset_index()
        )

        summary["missing_pct"] = (
            100
            * summary["n_missing"]
            / summary["n_obs"]
        )

        return summary.sort_values(
            "missing_pct",
            ascending=False
        )

    # ==================================================
    # RAW COVERAGE THROUGH TIME
    # ==================================================

    def raw_feature_coverage(self):
        """
        Analyse how often each feature is present across all
        historical cutoff snapshots.
        """

        if self.raw_features is None:
            raise ValueError(
                "raw_features.pkl is not available."
            )

        rows = []

        total_cutoffs = len(
            self.raw_features
        )

        for cutoff_date, snapshot in self.raw_features.items():

            available_series = set(
                snapshot[
                    "series_id"
                ].unique()
            )

            for series_id in available_series:

                df_series = snapshot[
                    snapshot["series_id"]
                    == series_id
                ]

                rows.append({
                    "cutoff_date":
                        pd.Timestamp(
                            cutoff_date
                        ),
                    "series_id":
                        series_id,
                    "feature_name":
                        df_series[
                            "feature_name"
                        ].iloc[0],
                    "n_obs":
                        len(df_series),
                    "n_missing":
                        df_series[
                            "value"
                        ].isna().sum()
                })

        coverage_long = pd.DataFrame(
            rows
        )

        coverage = (
            coverage_long
            .groupby(
                [
                    "series_id",
                    "feature_name"
                ]
            )
            .agg(
                cutoffs_available=(
                    "cutoff_date",
                    "nunique"
                ),
                first_cutoff=(
                    "cutoff_date",
                    "min"
                ),
                last_cutoff=(
                    "cutoff_date",
                    "max"
                ),
                total_observations=(
                    "n_obs",
                    "sum"
                ),
                total_missing=(
                    "n_missing",
                    "sum"
                )
            )
            .reset_index()
        )

        coverage["cutoff_coverage_pct"] = (
            100
            * coverage[
                "cutoffs_available"
            ]
            / total_cutoffs
        )

        coverage["raw_missing_pct"] = (
            100
            * coverage[
                "total_missing"
            ]
            / coverage[
                "total_observations"
            ]
        )

        return coverage.sort_values(
            "cutoff_coverage_pct"
        )

    # ==================================================
    # PROCESSED FEATURES - MISSING VALUES
    # ==================================================

    def processed_missing_summary(self):
        """
        Missingness of final engineered features.
        """

        if self.processed_features is None:
            raise ValueError(
                "processed_features.csv "
                "is not available."
            )

        df = (
            self.processed_features
            .copy()
        )

        # Remove cutoff date from feature analysis
        if "cutoff_date" in df.columns:
            df = df.drop(
                columns=["cutoff_date"]
            )

        result = pd.DataFrame({
            "feature": df.columns,
            "n_missing":
                df.isna().sum().values,
            "missing_pct":
                (
                    100
                    * df.isna().mean()
                ).values
        })

        return result.sort_values(
            "missing_pct",
            ascending=False
        ).reset_index(drop=True)

    # ==================================================
    # PROCESSED FEATURES - DISTRIBUTION
    # ==================================================

    def processed_distribution_summary(self):
        """
        Statistical distribution of engineered features.
        """

        if self.processed_features is None:
            raise ValueError(
                "processed_features.csv "
                "is not available."
            )

        df = (
            self.processed_features
            .copy()
        )

        if "cutoff_date" in df.columns:
            df = df.drop(
                columns=["cutoff_date"]
            )

        numeric_df = (
            df.select_dtypes(
                include=np.number
            )
        )

        summary = (
            numeric_df
            .describe(
                percentiles=[
                    0.01,
                    0.05,
                    0.25,
                    0.50,
                    0.75,
                    0.95,
                    0.99
                ]
            )
            .T
        )

        summary["missing_pct"] = (
            100
            * numeric_df.isna().mean()
        )

        summary["skewness"] = (
            numeric_df.skew()
        )

        summary["kurtosis"] = (
            numeric_df.kurtosis()
        )

        return summary

    # ==================================================
    # ZERO VARIANCE / VERY LOW VARIANCE
    # ==================================================

    def low_variance_features(
        self,
        threshold=1e-10
    ):

        if self.processed_features is None:
            raise ValueError(
                "processed_features.csv "
                "is not available."
            )

        df = (
            self.processed_features
            .select_dtypes(
                include=np.number
            )
        )

        variances = df.var()

        result = pd.DataFrame({
            "feature":
                variances.index,
            "variance":
                variances.values
        })

        return (
            result[
                result["variance"]
                <= threshold
            ]
            .sort_values(
                "variance"
            )
            .reset_index(
                drop=True
            )
        )

    # ==================================================
    # HIGH CORRELATIONS
    # ==================================================

    def high_correlations(
        self,
        threshold=0.85
    ):
        """
        Find highly correlated engineered features.
        """

        if self.processed_features is None:
            raise ValueError(
                "processed_features.csv "
                "is not available."
            )

        df = (
            self.processed_features
            .select_dtypes(
                include=np.number
            )
        )

        corr = df.corr()

        results = []

        columns = corr.columns

        for i in range(
            len(columns)
        ):

            for j in range(
                i + 1,
                len(columns)
            ):

                value = corr.iloc[
                    i,
                    j
                ]

                if (
                    pd.notna(value)
                    and
                    abs(value)
                    >= threshold
                ):

                    results.append({
                        "feature_1":
                            columns[i],
                        "feature_2":
                            columns[j],
                        "correlation":
                            value
                    })

        result = pd.DataFrame(
            results
        )

        if result.empty:
            return result

        result["abs_correlation"] = (
            result[
                "correlation"
            ].abs()
        )

        return (
            result
            .sort_values(
                "abs_correlation",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

    # ==================================================
    # DIMENSIONALITY
    # ==================================================

    def dimensionality_summary(self):
        """
        Compare sample size with number of engineered features.
        """

        if self.processed_features is None:
            raise ValueError(
                "processed_features.csv "
                "is not available."
            )

        df = (
            self.processed_features
            .copy()
        )

        feature_cols = [
            col
            for col in df.columns
            if col != "cutoff_date"
        ]

        n_observations = len(df)

        n_features = len(
            feature_cols
        )

        return pd.DataFrame([
            {
                "n_observations":
                    n_observations,
                "n_features":
                    n_features,
                "observations_per_feature":
                    (
                        n_observations
                        / n_features
                        if n_features > 0
                        else np.nan
                    )
            }
        ])

    # ==================================================
    # FINAL DATASET / TARGET
    # ==================================================

    def target_summary(
        self,
        target_col="target"
    ):

        if self.final_df is None:
            raise ValueError(
                "final_df.csv "
                "is not available."
            )

        if target_col not in self.final_df.columns:
            raise ValueError(
                f"{target_col} not found."
            )

        target = (
            pd.to_numeric(
                self.final_df[
                    target_col
                ],
                errors="coerce"
            )
        )

        summary = {
            "n_observations":
                len(target),
            "n_missing":
                target.isna().sum(),
            "missing_pct":
                100 * target.isna().mean(),
            "mean":
                target.mean(),
            "std":
                target.std(),
            "min":
                target.min(),
            "median":
                target.median(),
            "max":
                target.max(),
            "skewness":
                target.skew(),
            "kurtosis":
                target.kurtosis()
        }

        return pd.DataFrame(
            [summary]
        )

    # ==================================================
    # TARGET EXTREME VALUES
    # ==================================================

    def target_extremes(
        self,
        target_col="target",
        n=10
    ):

        if self.final_df is None:
            raise ValueError(
                "final_df.csv "
                "is not available."
            )

        columns = [
            col
            for col in [
                "cutoff_date",
                target_col
            ]
            if col in self.final_df.columns
        ]

        df = (
            self.final_df[
                columns
            ]
            .copy()
        )

        return pd.concat([
            df.nsmallest(
                n,
                target_col
            ),
            df.nlargest(
                n,
                target_col
            )
        ]).drop_duplicates()

    # ==================================================
    # PLOT FEATURE DISTRIBUTION
    # ==================================================

    def plot_feature_distribution(
        self,
        feature
    ):
        """
        Plot historical distribution of one processed feature.
        """

        if self.processed_features is None:
            raise ValueError(
                "processed_features.csv "
                "is not available."
            )

        if feature not in self.processed_features.columns:
            raise ValueError(
                f"Feature '{feature}' "
                "does not exist."
            )

        values = pd.to_numeric(
            self.processed_features[
                feature
            ],
            errors="coerce"
        ).dropna()

        plt.figure(
            figsize=(8, 5)
        )

        plt.hist(
            values,
            bins=20
        )

        plt.title(
            f"Distribution of {feature}"
        )

        plt.xlabel(feature)
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    # ==================================================
    # PLOT ALL FEATURE DISTRIBUTIONS
    # ==================================================

    def plot_all_feature_distributions(
        self
    ):
        """
        One independent histogram per processed feature.
        """

        if self.processed_features is None:
            raise ValueError(
                "processed_features.csv "
                "is not available."
            )

        numeric_cols = (
            self.processed_features
            .select_dtypes(
                include=np.number
            )
            .columns
        )

        for feature in numeric_cols:

            self.plot_feature_distribution(
                feature
            )
            
            
    
        
        