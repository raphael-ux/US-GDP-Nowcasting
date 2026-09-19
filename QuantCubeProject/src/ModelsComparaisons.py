import pandas as pd
from pathlib import Path

import TrainingModels
import AnalyseModels


class ModelsComparaisons:
    """
    Train several forecasting models and build a common
    performance comparison table.

    Workflow
    --------
    1. Train and tune every model.
    2. Generate out-of-fold predictions.
    3. Compute train / OOF / test metrics.
    4. Build one presentation-ready comparison table.
    """

    def __init__(
        self,
        model_types,
        X_train,
        y_train,
        X_test,
        y_test,
        dates_train,
        label_type="annualized_log_growth"
    ):

        # =====================================================
        # PROJECT PATHS
        # =====================================================

        self.project_root = (
            Path(__file__)
            .resolve()
            .parents[1]
        )

        self.comparison_path = (
            self.project_root
            / "data"
            / "models"
            / "models_comparison.csv"
        )

        # =====================================================
        # DATA
        # =====================================================

        self.model_types = model_types

        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        self.dates_train = dates_train

        self.label_type = label_type

        # =====================================================
        # RESULTS
        # =====================================================

        self.metrics_df = None
        self.comparison_df = None


    # =========================================================
    # TRAIN ALL MODELS
    # =========================================================

    def train_all(
        self,
        n_trials=50,
        n_splits=5,
        pca_bool=True
    ):
        """
        Train and tune every model.

        After fitting the best model, create OOF predictions
        so that AnalyseModels can compute OOF metrics.
        """

        for model in self.model_types:
            
            if model == "dumb_model":
                continue

            print("\n" + "=" * 60)

            print(
                f"TRAINING MODEL: "
                f"{model.upper()}"
            )

            print("=" * 60)

            # -------------------------------------------------
            # Create trainer
            # -------------------------------------------------

            trainer = (
                TrainingModels
                .TrainingModels(
                    X_train=self.X_train,
                    y_train=self.y_train,
                    model_type=model,
                    n_splits=n_splits,
                    dates_train=self.dates_train,
                    random_state=42,
                    pca_bool=pca_bool
                )
            )

            # -------------------------------------------------
            # Tune and fit best model
            # -------------------------------------------------

            trainer.tune(
                n_trials=n_trials
            )

            # -------------------------------------------------
            # Create OOF predictions
            # -------------------------------------------------

            trainer.create_oof_predictions()

        print(
            "\n"
            + "=" * 60
        )

        print(
            "ALL MODELS TRAINED"
        )

        print(
            "=" * 60
        )


    # =========================================================
    # ANALYSE ALL MODELS
    # =========================================================

    def analyse_all(self):
        """
        Compute train, out-of-fold and test metrics
        for every fitted model.

        Returns
        -------
        pd.DataFrame
        """

        all_metrics = []

        for model in self.model_types:

            print(
                f"Analysing {model}..."
            )

            # -------------------------------------------------
            # Create analysis object
            # -------------------------------------------------

            analysis = (
                AnalyseModels
                .AnalyseModels(
                    X_train=self.X_train,
                    y_train=self.y_train,
                    X_test=self.X_test,
                    y_test=self.y_test,
                    model_type=model,
                    label_type=self.label_type
                )
            )

            # -------------------------------------------------
            # Compute metrics
            # -------------------------------------------------

            metrics_df = (
                analysis.metrics()
            )

            all_metrics.append(
                metrics_df
            )

        # -----------------------------------------------------
        # Combine all models
        # -----------------------------------------------------

        self.metrics_df = pd.concat(
            all_metrics,
            ignore_index=True
        )

        return self.metrics_df


    # =========================================================
    # BUILD COMPARISON TABLE
    # =========================================================

    def build_comparison_df(self):
        """
        Build one row per model with:

        - Train metrics
        - OOF metrics
        - Test metrics

        Models are sorted by OOF RMSE.

        Returns
        -------
        pd.DataFrame
        """

        # -----------------------------------------------------
        # Compute metrics if not already done
        # -----------------------------------------------------

        if self.metrics_df is None:

            self.analyse_all()

        metrics = (
            self.metrics_df
            .copy()
        )

        # -----------------------------------------------------
        # Long format:
        #
        # ridge | train     | ...
        # ridge | OOF_train | ...
        # ridge | test      | ...
        #
        #             ↓
        #
        # One row per model
        # -----------------------------------------------------

        comparison = (
            metrics
            .pivot(
                index="model",
                columns="sample",
                values=[
                    "MAE",
                    "RMSE",
                    "R2",
                    "MAPE_pct"
                ]
            )
        )

        # -----------------------------------------------------
        # Flatten MultiIndex columns
        #
        # ('RMSE', 'OOF_train')
        #
        # becomes:
        #
        # OOF_train_RMSE
        # -----------------------------------------------------

        comparison.columns = [

            f"{sample}_{metric}"

            for metric, sample
            in comparison.columns
        ]

        comparison = (
            comparison
            .reset_index()
        )

        # -----------------------------------------------------
        # Rename columns for presentation
        # -----------------------------------------------------

        comparison = comparison.rename(
            columns={

                "model":
                    "Model",

                "train_MAE":
                    "Train MAE",

                "train_RMSE":
                    "Train RMSE",

                "train_R2":
                    "Train R2",

                "OOF_train_MAE":
                    "OOF MAE",

                "OOF_train_RMSE":
                    "OOF RMSE",

                "OOF_train_R2":
                    "OOF R2",

                "test_MAE":
                    "Test MAE",

                "test_RMSE":
                    "Test RMSE",

                "test_R2":
                    "Test R2",

                "train_MAPE_pct":
                    "Train MAPE",

                "OOF_train_MAPE_pct":
                    "OOF MAPE",

                "test_MAPE_pct":
                    "Test MAPE"
            }
        )

        # -----------------------------------------------------
        # Columns to show
        #
        # We intentionally do not emphasize MAPE because
        # GDP growth can be close to zero.
        # -----------------------------------------------------

        display_columns = [

            "Model",

            "Train MAE",
            "Train RMSE",
            "Train R2",

            "OOF MAE",
            "OOF RMSE",
            "OOF R2",

            "Test MAE",
            "Test RMSE",
            "Test R2"
        ]

        # Keep only columns that actually exist
        display_columns = [

            column

            for column
            in display_columns

            if column
            in comparison.columns
        ]

        comparison = comparison[
            display_columns
        ]

        # -----------------------------------------------------
        # Sort by OOF RMSE
        #
        # This is the metric used for model comparison,
        # NOT test performance.
        # -----------------------------------------------------

        if "OOF RMSE" in comparison.columns:

            comparison = (
                comparison
                .sort_values(
                    "OOF RMSE",
                    ascending=True
                )
                .reset_index(
                    drop=True
                )
            )

        # -----------------------------------------------------
        # Round metrics
        # -----------------------------------------------------

        numeric_columns = (
            comparison
            .select_dtypes(
                include="number"
            )
            .columns
        )

        comparison[
            numeric_columns
        ] = (

            comparison[
                numeric_columns
            ]

            .round(3)
        )

        # -----------------------------------------------------
        # Save result
        # -----------------------------------------------------

        self.comparison_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        comparison.to_csv(
            self.comparison_path,
            index=False
        )

        self.comparison_df = (
            comparison
        )

        print(
            f"\nComparison table saved to:"
            f"\n{self.comparison_path}"
        )

        return self.comparison_df


    # =========================================================
    # RUN EVERYTHING
    # =========================================================

    def run_all(
        self,
        n_trials=50,
        n_splits=5,
        pca_bool=False
    ):
        """
        Convenience method:

        train
        → analyse
        → build final comparison table
        """

        self.train_all(
            n_trials=n_trials,
            n_splits=n_splits,
            pca_bool=pca_bool
        )

        self.analyse_all()

        return (
            self.build_comparison_df()
        )
        
        
    def compare_pca(
        self,
        n_trials=50,
        n_splits=5
    ):

        all_results = []

        linear_models = [
            "linear",
            "ridge",
            "lasso",
            "elasticnet"
        ]

        for model in self.model_types:

            # ==============================================
            # DUMB MODEL
            # ==============================================

            if model == "dumb_model":

                analysis = AnalyseModels.AnalyseModels(
                    X_train=self.X_train,
                    y_train=self.y_train,
                    X_test=self.X_test,
                    y_test=self.y_test,
                    model_type="dumb_model",
                    label_type=self.label_type,
                    n_splits=n_splits
                )

                metrics = analysis.metrics()

                metrics["PCA"] = "N/A"

                all_results.append(
                    metrics
                )

                continue

            # ==============================================
            # PCA OPTIONS
            # ==============================================

            if model in linear_models:

                pca_options = [
                    False,
                    True
                ]

            else:

                # XGB / CatBoost:
                # PCA comparison is not necessary
                pca_options = [
                    False
                ]

            # ==============================================
            # TRAIN EACH CONFIGURATION
            # ==============================================

            for pca_bool in pca_options:

                print("\n" + "=" * 70)

                print(
                    f"MODEL: {model.upper()} "
                    f"| PCA: {pca_bool}"
                )

                print("=" * 70)

                # ------------------------------------------
                # TRAIN
                # ------------------------------------------

                trainer = (
                    TrainingModels.TrainingModels(
                        X_train=self.X_train,
                        y_train=self.y_train,
                        model_type=model,
                        n_splits=n_splits,
                        dates_train=self.dates_train,
                        random_state=42,
                        pca_bool=pca_bool
                    )
                )

                trainer.tune(
                    n_trials=n_trials
                )

                trainer.create_oof_predictions()

                # ------------------------------------------
                # ANALYSE IMMEDIATELY
                #
                # Important because next PCA configuration
                # may overwrite the saved model.
                # ------------------------------------------

                analysis = (
                    AnalyseModels.AnalyseModels(
                        X_train=self.X_train,
                        y_train=self.y_train,
                        X_test=self.X_test,
                        y_test=self.y_test,
                        model_type=model,
                        label_type=self.label_type,
                        n_splits=n_splits
                    )
                )

                metrics = (
                    analysis.metrics()
                    .copy()
                )

                metrics["PCA"] = (
                    "Yes"
                    if pca_bool
                    else "No"
                )

                all_results.append(
                    metrics
                )

        # ==============================================
        # CONCATENATE
        # ==============================================

        self.pca_metrics_df = (
            pd.concat(
                all_results,
                ignore_index=True
            )
        )

        return self.pca_metrics_df
    
    def build_pca_comparison_df(self):

        if not hasattr(
            self,
            "pca_metrics_df"
        ):

            raise ValueError(
                "Run compare_pca() first."
            )

        df = (
            self.pca_metrics_df
            .copy()
        )

        # ==============================================
        # PIVOT
        # ==============================================

        comparison = (
            df.pivot_table(
                index=[
                    "model",
                    "PCA"
                ],
                columns="sample",
                values=[
                    "MAE",
                    "RMSE",
                    "R2"
                ]
            )
        )

        # Flatten MultiIndex columns
        comparison.columns = [

            f"{sample}_{metric}"

            for metric, sample
            in comparison.columns

        ]

        comparison = (
            comparison
            .reset_index()
        )

        # ==============================================
        # RENAME
        # ==============================================

        comparison = (
            comparison.rename(
                columns={
                    "train_MAE":
                        "Train MAE",

                    "train_RMSE":
                        "Train RMSE",

                    "train_R2":
                        "Train R2",

                    "OOF_train_MAE":
                        "OOF MAE",

                    "OOF_train_RMSE":
                        "OOF RMSE",

                    "OOF_train_R2":
                        "OOF R2",

                    "test_MAE":
                        "Test MAE",

                    "test_RMSE":
                        "Test RMSE",

                    "test_R2":
                        "Test R2"
                }
            )
        )

        # ==============================================
        # ORDER COLUMNS
        # ==============================================

        columns = [
            "model",
            "PCA",
            "Train MAE",
            "Train RMSE",
            "Train R2",
            "OOF MAE",
            "OOF RMSE",
            "OOF R2",
            "Test MAE",
            "Test RMSE",
            "Test R2"
        ]

        comparison = (
            comparison[
                columns
            ]
        )

        comparison = (
            comparison.round(3)
        )

        # ==============================================
        # SAVE
        # ==============================================

        output_path = (
            self.project_root
            / "data"
            / "models"
            / "pca_comparison.csv"
        )

        comparison.to_csv(
            output_path,
            index=False
        )

        self.pca_comparison_df = (
            comparison
        )

        return comparison