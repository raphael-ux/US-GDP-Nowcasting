import numpy as np
import pandas as pd
import optuna

from pathlib import Path
from optuna.samplers import TPESampler

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor


class TrainingModels:
    """
    Main class for model training and hyperparameter tuning.

    Supported models:
        - linear
        - ridge
        - lasso
        - elasticnet
        - xgb
        - catboost

    Workflow:
        1. Receive training data only.
        2. Create TimeSeriesSplit internally.
        3. Tune hyperparameters using Optuna.
        4. Select parameters minimizing mean validation RMSE.
        5. Refit the best model on the complete training set.
        6. Generate predictions on unseen data.
        7. Generate out-of-fold predictions on the training set.
    """

    def __init__(
        self,
        X_train,
        y_train,
        model_type,
        n_splits=5,
        dates_train=None,
        random_state=42,
        pca_bool = False
    ):
        self.pca_bool = pca_bool
        
        self.project_root = Path(__file__).resolve().parents[1]
        
        self.model_type = model_type.lower()
        
        self.oof_predictions_path = (
                    self.project_root
                    / "data"
                    / "models"
                    / self.model_type
                    / "oof_predictions.csv"
                )
        
        self.best_model_path = (
                    self.project_root
                    / "data"
                    / "models"
                    / self.model_type
                    / "best_model.pkl"
                )

        # --------------------------------------------------
        # Data
        # --------------------------------------------------

        self.X_train = X_train.copy()
        self.y_train = y_train.copy()

        self.dates_train = (
            dates_train.reset_index(drop=True)
            if dates_train is not None
            else None
        )

        # Reset indices to make TimeSeriesSplit / iloc clean
        self.X_train = self.X_train.reset_index(drop=True)
        self.y_train = self.y_train.reset_index(drop=True)

        # --------------------------------------------------
        # Model configuration
        # --------------------------------------------------

        self.n_splits = n_splits
        self.random_state = random_state

        # --------------------------------------------------
        # Time-series cross-validation
        # --------------------------------------------------

        self.cv = TimeSeriesSplit(
            n_splits=self.n_splits
        )

        # --------------------------------------------------
        # Results
        # --------------------------------------------------

        self.study = None

        self.best_params = None
        self.best_model = None

        self.best_cv_rmse = None
        self.best_cv_rmse_std = None

        self.oof_predictions = None

        # --------------------------------------------------
        # Validate model type
        # --------------------------------------------------

        supported_models = [
            "linear",
            "ridge",
            "lasso",
            "elasticnet",
            "xgb",
            "catboost"
        ]

        if self.model_type not in supported_models:

            raise ValueError(
                f"model_type must be one of "
                f"{supported_models}"
            )

        # --------------------------------------------------
        # Basic data checks
        # --------------------------------------------------

        if len(self.X_train) != len(self.y_train):

            raise ValueError(
                "X_train and y_train must have "
                "the same number of observations."
            )

        if self.dates_train is not None:

            if len(self.dates_train) != len(self.X_train):

                raise ValueError(
                    "dates_train must have the same "
                    "length as X_train."
                )


    # =========================================================
    # HYPERPARAMETER SEARCH SPACE
    # =========================================================

    def _sample_params(
        self,
        trial
    ):
        """
        Define Optuna hyperparameter search space
        depending on the selected model.
        """

        # --------------------------------------------------
        # Linear Regression
        # --------------------------------------------------

        if self.model_type == "linear":

            return {}

        # --------------------------------------------------
        # Ridge
        # --------------------------------------------------

        elif self.model_type == "ridge":

            return {

                "alpha": trial.suggest_float(
                    "alpha",
                    1e-4,
                    1e4,
                    log=True
                )
            }

        # --------------------------------------------------
        # Lasso
        # --------------------------------------------------

        elif self.model_type == "lasso":

            return {

                "alpha": trial.suggest_float(
                    "alpha",
                    1e-5,
                    1e2,
                    log=True
                )
            }

        # --------------------------------------------------
        # Elastic Net
        # --------------------------------------------------

        elif self.model_type == "elasticnet":

            return {

                "alpha": trial.suggest_float(
                    "alpha",
                    1e-5,
                    1e2,
                    log=True
                ),

                "l1_ratio": trial.suggest_float(
                    "l1_ratio",
                    0.0,
                    1.0
                )
            }

        # --------------------------------------------------
        # XGBoost
        # --------------------------------------------------

        elif self.model_type == "xgb":

            return {

                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    50,
                    500
                ),

                "max_depth": trial.suggest_int(
                    "max_depth",
                    1,
                    5
                ),

                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.2,
                    log=True
                ),

                "subsample": trial.suggest_float(
                    "subsample",
                    0.6,
                    1.0
                ),

                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree",
                    0.5,
                    1.0
                ),

                "min_child_weight": trial.suggest_int(
                    "min_child_weight",
                    1,
                    10
                ),

                "reg_alpha": trial.suggest_float(
                    "reg_alpha",
                    1e-8,
                    10,
                    log=True
                ),

                "reg_lambda": trial.suggest_float(
                    "reg_lambda",
                    1e-8,
                    10,
                    log=True
                )
            }

        # --------------------------------------------------
        # CatBoost
        # --------------------------------------------------

        elif self.model_type == "catboost":

            return {

                "iterations": trial.suggest_int(
                    "iterations",
                    100,
                    800
                ),

                "depth": trial.suggest_int(
                    "depth",
                    2,
                    6
                ),

                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.2,
                    log=True
                ),

                "l2_leaf_reg": trial.suggest_float(
                    "l2_leaf_reg",
                    1e-3,
                    10,
                    log=True
                )
            }


    # =========================================================
    # MODEL CREATION
    # =========================================================

    def _build_model(
        self,
        params
    ):
        """
        Build estimator from parameter dictionary.
        """

        if self.model_type == "linear":

            return LinearRegression()

        elif self.model_type == "ridge":

            return Ridge(
                **params
            )

        elif self.model_type == "lasso":

            return Lasso(
                **params,
                max_iter=10000
            )

        elif self.model_type == "elasticnet":

            return ElasticNet(
                **params,
                max_iter=20000,
                tol=1e-4,
                random_state=self.random_state
            )

        elif self.model_type == "xgb":

            return XGBRegressor(
                **params,
                objective="reg:squarederror",
                random_state=self.random_state,
                n_jobs=-1
            )

        elif self.model_type == "catboost":

            try:
                from catboost import CatBoostRegressor

            except ImportError:

                raise ImportError(
                    "CatBoost is not installed. "
                    "Install it with: pip install catboost"
                )

            return CatBoostRegressor(
                **params,
                loss_function="RMSE",
                random_seed=self.random_state,
                verbose=False
            )


    # =========================================================
    # PIPELINE
    # =========================================================

    def _build_pipeline(
        self,
        params
    ):
        """
        Build preprocessing + model pipeline.

        Linear models:
            median imputation
                ->
            standardization
                ->
            optional PCA
                ->
            regression

        Tree models:
            median imputation
                ->
            regression
        """

        model = self._build_model(
            params
        )

        steps = [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True
                )
            )
        ]

        # --------------------------------------------------
        # Linear models
        # --------------------------------------------------

        if self.model_type in [
            "linear",
            "ridge",
            "lasso",
            "elasticnet"
        ]:

            steps.append(
                (
                    "scaler",
                    StandardScaler()
                )
            )

            # PCA only after standardization
            if self.pca_bool:

                steps.append(
                    (
                        "pca",
                        PCA(
                            n_components=0.90
                        )
                    )
                )

        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        steps.append(
            (
                "model",
                model
            )
        )

        return Pipeline(
            steps
    )


    # =========================================================
    # OPTUNA OBJECTIVE
    # =========================================================

    def _objective(
        self,
        trial
    ):
        """
        Optuna objective.

        One trial corresponds to one hyperparameter
        combination.

        The same parameters are tested on every
        TimeSeriesSplit fold.

        Objective:
            minimize mean validation RMSE.
        """

        params = self._sample_params(
            trial
        )

        fold_rmse = []

        # --------------------------------------------------
        # Time-series folds
        # --------------------------------------------------

        for train_idx, val_idx in self.cv.split(
            self.X_train
        ):

            X_tr = self.X_train.iloc[
                train_idx
            ]

            X_val = self.X_train.iloc[
                val_idx
            ]

            y_tr = self.y_train.iloc[
                train_idx
            ]

            y_val = self.y_train.iloc[
                val_idx
            ]

            # ----------------------------------------------
            # New independent pipeline for this fold
            # ----------------------------------------------

            pipeline = self._build_pipeline(
                params
            )

            pipeline.fit(
                X_tr,
                y_tr
            )

            predictions = pipeline.predict(
                X_val
            )

            # ----------------------------------------------
            # Fold RMSE
            # ----------------------------------------------

            rmse = np.sqrt(

                mean_squared_error(
                    y_val,
                    predictions
                )
            )

            fold_rmse.append(
                rmse
            )

        # --------------------------------------------------
        # Aggregate folds
        # --------------------------------------------------

        mean_rmse = float(
            np.mean(
                fold_rmse
            )
        )

        rmse_std = float(
            np.std(
                fold_rmse
            )
        )

        trial.set_user_attr(
            "fold_rmse",
            fold_rmse
        )

        trial.set_user_attr(
            "rmse_std",
            rmse_std
        )

        return mean_rmse


    # =========================================================
    # MODELS WITHOUT HYPERPARAMETER TUNING
    # =========================================================

    def _evaluate_without_tuning(
        self
    ):
        """
        Evaluate models with no hyperparameters to tune,
        such as ordinary Linear Regression.
        """

        fold_rmse = []

        for train_idx, val_idx in self.cv.split(
            self.X_train
        ):

            X_tr = self.X_train.iloc[
                train_idx
            ]

            X_val = self.X_train.iloc[
                val_idx
            ]

            y_tr = self.y_train.iloc[
                train_idx
            ]

            y_val = self.y_train.iloc[
                val_idx
            ]

            pipeline = self._build_pipeline(
                params={}
            )

            pipeline.fit(
                X_tr,
                y_tr
            )

            predictions = pipeline.predict(
                X_val
            )

            rmse = np.sqrt(

                mean_squared_error(
                    y_val,
                    predictions
                )
            )

            fold_rmse.append(
                rmse
            )

        self.best_cv_rmse = float(
            np.mean(
                fold_rmse
            )
        )

        self.best_cv_rmse_std = float(
            np.std(
                fold_rmse
            )
        )

        return fold_rmse


    # =========================================================
    # TUNING
    # =========================================================

    def tune(
        self,
        n_trials=20
    ):
        """
        Tune hyperparameters using TimeSeriesSplit.

        Once the best parameters have been selected,
        refit the model on the entire training dataset.
        """

        # --------------------------------------------------
        # Linear Regression
        # --------------------------------------------------

        if self.model_type == "linear":

            self.best_params = {}

            self._evaluate_without_tuning()

        # --------------------------------------------------
        # Models requiring Optuna
        # --------------------------------------------------

        else:

            optuna.logging.set_verbosity(
                optuna.logging.WARNING
            )

            self.study = optuna.create_study(

                direction="minimize",

                sampler=TPESampler(
                    seed=self.random_state
                )
            )

            self.study.optimize(

                self._objective,

                n_trials=n_trials,

                show_progress_bar=True
            )

            # ----------------------------------------------
            # Best hyperparameters
            # ----------------------------------------------

            self.best_params = dict(
                self.study.best_params
            )

            self.best_cv_rmse = float(
                self.study.best_value
            )

            self.best_cv_rmse_std = float(

                self.study
                .best_trial
                .user_attrs[
                    "rmse_std"
                ]
            )

        # --------------------------------------------------
        # Fit best model on ALL training observations
        # --------------------------------------------------

        self.best_model = (
            self._build_pipeline(
                self.best_params
            )
        )

        self.best_model.fit(
            self.X_train,
            self.y_train
        )
        
        import joblib
        
        joblib.dump(
            self.best_model,
            self.best_model_path
        )

        print(
            f"Best model saved to: "
            f"{self.best_model_path}"
        )
        
        # --------------------------------------------------
        # Summary
        # --------------------------------------------------

        print(
            f"\nModel: {self.model_type}"
        )

        print(
            f"CV RMSE: "
            f"{self.best_cv_rmse:.4f} "
            f"+/- "
            f"{self.best_cv_rmse_std:.4f}"
        )

        print(
            f"Best parameters: "
            f"{self.best_params}"
        )

        return self.best_model


    # =========================================================
    # OUT-OF-FOLD PREDICTIONS
    # =========================================================

    def create_oof_predictions(
        self
    ):
        """
        Generate out-of-fold predictions using the best
        hyperparameters.

        For each TimeSeriesSplit fold:

            train on past observations
                    ↓
            predict validation observations

        The validation predictions are concatenated into
        one DataFrame.

        Parameters
        ----------
        save_path : str or Path, optional
            If provided, save the OOF DataFrame as CSV.

        Returns
        -------
        pd.DataFrame
        """

        if self.best_params is None:

            raise ValueError(
                "Call tune() before "
                "create_oof_predictions()."
            )

        oof_rows = []

        # --------------------------------------------------
        # Replay CV using best parameters
        # --------------------------------------------------

        for fold, (
            train_idx,
            val_idx
        ) in enumerate(

            self.cv.split(
                self.X_train
            ),

            start=1
        ):

            # ----------------------------------------------
            # Training fold
            # ----------------------------------------------

            X_tr = self.X_train.iloc[
                train_idx
            ]

            y_tr = self.y_train.iloc[
                train_idx
            ]

            # ----------------------------------------------
            # Validation fold
            # ----------------------------------------------

            X_val = self.X_train.iloc[
                val_idx
            ]

            y_val = self.y_train.iloc[
                val_idx
            ]

            # ----------------------------------------------
            # New model with best parameters
            # ----------------------------------------------

            pipeline = self._build_pipeline(
                self.best_params
            )

            pipeline.fit(
                X_tr,
                y_tr
            )

            predictions = pipeline.predict(
                X_val
            )

            # ----------------------------------------------
            # Fold output
            # ----------------------------------------------

            fold_df = pd.DataFrame({

                "row_index": val_idx,

                "y_true": y_val.values,

                "y_pred": predictions,

                "fold": fold
            })

            # ----------------------------------------------
            # Add date if available
            # ----------------------------------------------

            if self.dates_train is not None:

                fold_df[
                    "cutoff_date"
                ] = (

                    self.dates_train
                    .iloc[val_idx]
                    .values
                )

            oof_rows.append(
                fold_df
            )

        # --------------------------------------------------
        # Concatenate validation predictions
        # --------------------------------------------------

        oof_df = pd.concat(
            oof_rows,
            ignore_index=True
        )

        # --------------------------------------------------
        # Errors
        # --------------------------------------------------

        oof_df["error"] = (

            oof_df["y_true"]
            - oof_df["y_pred"]
        )

        oof_df["absolute_error"] = (
            oof_df["error"].abs()
        )

        oof_df["squared_error"] = (
            oof_df["error"] ** 2
        )

        # --------------------------------------------------
        # Sort
        # --------------------------------------------------

        if "cutoff_date" in oof_df.columns:

            oof_df[
                "cutoff_date"
            ] = pd.to_datetime(
                oof_df["cutoff_date"]
            )

            oof_df = (

                oof_df
                .sort_values(
                    "cutoff_date"
                )
                .reset_index(
                    drop=True
                )
            )

        else:

            oof_df = (

                oof_df
                .sort_values(
                    "row_index"
                )
                .reset_index(
                    drop=True
                )
            )

        # --------------------------------------------------
        # Store internally
        # --------------------------------------------------

        self.oof_predictions = oof_df

        # --------------------------------------------------
        # Optional saving
        # --------------------------------------------------



        self.oof_predictions.to_csv(
            self.oof_predictions_path,
            index=False
        )

        print(
            f"OOF predictions saved to: "
            f"{self.oof_predictions_path}"
        )

        return self.oof_predictions


    # =========================================================
    # PREDICTION ON NEW / TEST DATA
    # =========================================================

    def predict(
        self,
        X
    ):
        """
        Predict using the final model fitted on the full
        training dataset.
        """

        if self.best_model is None:

            raise ValueError(
                "Call tune() before predict()."
            )

        return self.best_model.predict(
            X
        )


    # =========================================================
    # GET BEST PARAMETERS
    # =========================================================

    def get_best_params(
        self
    ):
        """
        Return best hyperparameters.
        """

        if self.best_params is None:

            raise ValueError(
                "Call tune() first."
            )

        return self.best_params


    # =========================================================
    # GET CV RESULTS
    # =========================================================

    def get_cv_score(
        self
    ):
        """
        Return best cross-validation RMSE.
        """

        if self.best_cv_rmse is None:

            raise ValueError(
                "Call tune() first."
            )

        return {

            "rmse_mean":
                self.best_cv_rmse,

            "rmse_std":
                self.best_cv_rmse_std
        }
        
        
    
        