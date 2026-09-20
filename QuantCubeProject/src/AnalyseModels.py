from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

from sklearn.model_selection import TimeSeriesSplit


class AnalyseModels:

    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        model_type,
        label_type=None,
        dates_train=None,
        dates_test=None,
        n_splits=5
    ):
        self.project_root = (
            Path(__file__)
            .resolve()
            .parents[1]
        )

        self.model_type = (
            model_type.lower()
        )

        if label_type:

            self.label_type = (
                label_type.lower()
            )

        else:

            self.label_type = None

        self.n_splits = n_splits


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

        self.metrics_path = (
            self.project_root
            / "data"
            / "models"
            / self.model_type
            / "metrics.csv"
        )


        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test


        self.dates_train = (

            pd.to_datetime(
                dates_train
            ).reset_index(
                drop=True
            )

            if dates_train is not None

            else None
        )

        self.dates_test = (

            pd.to_datetime(
                dates_test
            ).reset_index(
                drop=True
            )

            if dates_test is not None

            else None
        )

        if (
            self.dates_train is not None
            and
            len(self.dates_train)
            != len(self.y_train)
        ):

            raise ValueError(
                "dates_train must have the "
                "same length as y_train."
            )

        if (
            self.dates_test is not None
            and
            len(self.dates_test)
            != len(self.y_test)
        ):

            raise ValueError(
                "dates_test must have the "
                "same length as y_test."
            )

        self.best_model = None

        self.oof_predictions = None

        self.metrics_df = None


    def _compute_metrics(
        self,
        y_true,
        y_pred
    ):

        mae = mean_absolute_error(
            y_true,
            y_pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )

        r2 = r2_score(
            y_true,
            y_pred
        )

        mape = (
            mean_absolute_percentage_error(
                y_true,
                y_pred
            )
            * 100
        )

        return {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE_pct": mape
        }


    def metrics(self):

        """
        Compute regression metrics for:

        1. Full training set
        2. Out-of-fold training observations
        3. Final untouched test set

        For dumb_model:

        - level target:
              prediction = previous GDP value

        - growth target:
              prediction = 0
        """

        if self.model_type == "dumb_model":

            cv = TimeSeriesSplit(
                n_splits=self.n_splits
            )


            if self.label_type == "level":


                train_predictions = (
                    self.y_train.shift(1)
                )

                train_mask = (
                    train_predictions.notna()
                )

                train_metrics = (
                    self._compute_metrics(
                        y_true=self.y_train[
                            train_mask
                        ],
                        y_pred=train_predictions[
                            train_mask
                        ]
                    )
                )

                lagged_y = (
                    self.y_train.shift(1)
                )

                oof_true = []

                oof_pred = []

                for (
                    _,
                    val_idx
                ) in cv.split(
                    self.X_train
                ):

                    fold_true = (
                        self.y_train
                        .iloc[val_idx]
                    )

                    fold_pred = (
                        lagged_y
                        .iloc[val_idx]
                    )

                    mask = (
                        fold_pred.notna()
                    )

                    oof_true.extend(
                        fold_true[
                            mask
                        ].to_numpy()
                    )

                    oof_pred.extend(
                        fold_pred[
                            mask
                        ].to_numpy()
                    )

                oof_metrics = (
                    self._compute_metrics(
                        y_true=np.asarray(
                            oof_true
                        ),
                        y_pred=np.asarray(
                            oof_pred
                        )
                    )
                )


                test_predictions = (
                    np.concatenate(
                        [
                            [
                                self.y_train
                                .iloc[-1]
                            ],
                            self.y_test
                            .iloc[:-1]
                            .to_numpy()
                        ]
                    )
                )

                test_metrics = (
                    self._compute_metrics(
                        y_true=self.y_test,
                        y_pred=test_predictions
                    )
                )


            elif self.label_type in [

                "difference",
                "pct_change",
                "log_growth",
                "annualized_log_growth"

            ]:


                train_predictions = (
                    np.zeros(
                        len(self.y_train)
                    )
                )

                train_metrics = (
                    self._compute_metrics(
                        y_true=self.y_train,
                        y_pred=train_predictions
                    )
                )


                oof_true = []

                for (
                    _,
                    val_idx
                ) in cv.split(
                    self.X_train
                ):

                    oof_true.extend(

                        self.y_train
                        .iloc[val_idx]
                        .to_numpy()

                    )

                oof_true = (
                    np.asarray(
                        oof_true
                    )
                )

                oof_predictions = (
                    np.zeros(
                        len(oof_true)
                    )
                )

                oof_metrics = (
                    self._compute_metrics(
                        y_true=oof_true,
                        y_pred=oof_predictions
                    )
                )



                test_predictions = (
                    np.zeros(
                        len(self.y_test)
                    )
                )

                test_metrics = (
                    self._compute_metrics(
                        y_true=self.y_test,
                        y_pred=test_predictions
                    )
                )

            else:

                raise ValueError(
                    "Unknown label_type: "
                    f"{self.label_type}"
                )


            self.metrics_df = (
                pd.DataFrame([
                    {
                        "model":
                            "dumb_model",

                        "sample":
                            "train",

                        **train_metrics
                    },
                    {
                        "model":
                            "dumb_model",

                        "sample":
                            "OOF_train",

                        **oof_metrics
                    },
                    {
                        "model":
                            "dumb_model",

                        "sample":
                            "test",

                        **test_metrics
                    }
                ])
            )

        else:


            if not self.best_model_path.exists():

                raise FileNotFoundError(
                    f"Best model not found at: "
                    f"{self.best_model_path}"
                )

            self.best_model = (
                joblib.load(
                    self.best_model_path
                )
            )


            if not self.oof_predictions_path.exists():

                raise FileNotFoundError(
                    f"OOF predictions not found at: "
                    f"{self.oof_predictions_path}"
                )

            self.oof_predictions = (
                pd.read_csv(
                    self.oof_predictions_path
                )
            )



            train_predictions = (
                self.best_model.predict(
                    self.X_train
                )
            )

            train_metrics = (
                self._compute_metrics(
                    y_true=self.y_train,
                    y_pred=train_predictions
                )
            )


            oof_metrics = (
                self._compute_metrics(
                    y_true=
                        self.oof_predictions[
                            "y_true"
                        ],

                    y_pred=
                        self.oof_predictions[
                            "y_pred"
                        ]
                )
            )


            test_predictions = (
                self.best_model.predict(
                    self.X_test
                )
            )

            test_metrics = (
                self._compute_metrics(
                    y_true=self.y_test,
                    y_pred=test_predictions
                )
            )


            self.metrics_df = (
                pd.DataFrame([
                    {
                        "model":
                            self.model_type,

                        "sample":
                            "train",

                        **train_metrics
                    },
                    {
                        "model":
                            self.model_type,

                        "sample":
                            "OOF_train",

                        **oof_metrics
                    },
                    {
                        "model":
                            self.model_type,

                        "sample":
                            "test",

                        **test_metrics
                    }
                ])
            )
            
        self.metrics_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.metrics_df.to_csv(
            self.metrics_path,
            index=False
        )

        return self.metrics_df


    def feature_coefficients(self):

        """
        Analyse coefficients for linear models.

        Without PCA:
            returns coefficients associated
            with original features.

        With PCA:
            returns:
            1. coefficients associated with
               principal components
            2. approximate contribution of
               original features reconstructed
               from PCA loadings
        """

        linear_models = [
            "linear",
            "ridge",
            "lasso",
            "elasticnet"
        ]

        if self.model_type not in linear_models:

            raise ValueError(
                "feature_coefficients() is only "
                "available for "
                f"{linear_models}."
            )


        if self.best_model is None:

            if not self.best_model_path.exists():

                raise FileNotFoundError(
                    f"Best model not found at: "
                    f"{self.best_model_path}"
                )

            self.best_model = (
                joblib.load(
                    self.best_model_path
                )
            )

        pipeline = (
            self.best_model
        )

        model = (
            pipeline.named_steps[
                "model"
            ]
        )

        coefficients = (
            np.asarray(
                model.coef_
            ).ravel()
        )


        if "pca" not in pipeline.named_steps:

            feature_names = (
                self.X_train
                .columns
                .tolist()
            )

            coefficients_df = (
                pd.DataFrame({
                    "feature":
                        feature_names,

                    "coefficient":
                        coefficients
                })
            )

            coefficients_df[
                "abs_coefficient"
            ] = (
                coefficients_df[
                    "coefficient"
                ].abs()
            )

            coefficients_df[
                "selected"
            ] = (
                coefficients_df[
                    "abs_coefficient"
                ] > 1e-12
            )

            coefficients_df = (
                coefficients_df
                .sort_values(
                    "abs_coefficient",
                    ascending=False
                )
                .reset_index(
                    drop=True
                )
            )

            coefficients_df[
                "rank"
            ] = (
                np.arange(
                    1,
                    len(
                        coefficients_df
                    ) + 1
                )
            )

            return coefficients_df[
                [
                    "rank",
                    "feature",
                    "coefficient",
                    "abs_coefficient",
                    "selected"
                ]
            ]


        else:

            pca = (
                pipeline.named_steps[
                    "pca"
                ]
            )


            component_names = [

                f"PC{i + 1}"

                for i in range(
                    len(
                        coefficients
                    )
                )
            ]

            component_df = (
                pd.DataFrame({
                    "component":
                        component_names,

                    "coefficient":
                        coefficients,

                    "explained_variance_ratio":
                        pca.explained_variance_ratio_
                })
            )

            component_df[
                "abs_coefficient"
            ] = (
                component_df[
                    "coefficient"
                ].abs()
            )

            component_df = (
                component_df
                .sort_values(
                    "abs_coefficient",
                    ascending=False
                )
                .reset_index(
                    drop=True
                )
            )


            original_coefficients = (
                pca.components_.T
                @ coefficients
            )

            feature_df = (
                pd.DataFrame({
                    "feature":
                        self.X_train.columns,

                    "reconstructed_coefficient":
                        original_coefficients
                })
            )

            feature_df[
                "abs_reconstructed_coefficient"
            ] = (
                feature_df[
                    "reconstructed_coefficient"
                ].abs()
            )

            feature_df = (
                feature_df
                .sort_values(
                    "abs_reconstructed_coefficient",
                    ascending=False
                )
                .reset_index(
                    drop=True
                )
            )

            feature_df[
                "rank"
            ] = (
                np.arange(
                    1,
                    len(feature_df) + 1
                )
            )

            return {
                "components":
                    component_df,

                "features":
                    feature_df
            }


    def pca_loadings(self):

        """
        Return PCA loadings for a fitted
        model using PCA.

        Rows:
            original features

        Columns:
            principal components
        """

        if self.best_model is None:

            if not self.best_model_path.exists():

                raise FileNotFoundError(
                    f"Best model not found at: "
                    f"{self.best_model_path}"
                )

            self.best_model = (
                joblib.load(
                    self.best_model_path
                )
            )

        if (
            "pca"
            not in self.best_model.named_steps
        ):

            raise ValueError(
                "This model does not use PCA."
            )

        pca = (
            self.best_model
            .named_steps[
                "pca"
            ]
        )

        loadings = pd.DataFrame(

            pca.components_.T,

            index=
                self.X_train.columns,

            columns=[
                f"PC{i + 1}"

                for i in range(
                    pca.n_components_
                )
            ]
        )

        return loadings


    def plot_predictions(self):

        """
        Plot actual labels versus predictions for:

        1. Full training sample
        2. Out-of-fold training predictions
        3. Final test sample
        """

        if self.model_type == "dumb_model":

            raise ValueError(
                "plot_predictions() is intended "
                "for trained models with saved "
                "OOF predictions."
            )


        if self.best_model is None:

            if not self.best_model_path.exists():

                raise FileNotFoundError(
                    f"Best model not found at: "
                    f"{self.best_model_path}"
                )

            self.best_model = (
                joblib.load(
                    self.best_model_path
                )
            )


        if self.oof_predictions is None:

            if not self.oof_predictions_path.exists():

                raise FileNotFoundError(
                    f"OOF predictions not found at: "
                    f"{self.oof_predictions_path}"
                )

            self.oof_predictions = (
                pd.read_csv(
                    self.oof_predictions_path
                )
            )


        train_predictions = (
            self.best_model.predict(
                self.X_train
            )
        )

        test_predictions = (
            self.best_model.predict(
                self.X_test
            )
        )

        if self.dates_train is not None:

            x_train = (
                self.dates_train
            )

            train_xlabel = "Date"

        else:

            x_train = (
                np.arange(
                    len(self.y_train)
                )
            )

            train_xlabel = (
                "Observation"
            )

        if self.dates_test is not None:

            x_test = (
                self.dates_test
            )

            test_xlabel = "Date"

        else:

            x_test = (
                np.arange(
                    len(self.y_test)
                )
            )

            test_xlabel = (
                "Observation"
            )


        plt.figure(
            figsize=(12, 5)
        )

        plt.plot(
            x_train,
            np.asarray(
                self.y_train
            ),
            label="Actual"
        )

        plt.plot(
            x_train,
            train_predictions,
            label="Prediction"
        )

        plt.axhline(
            0,
            linewidth=0.8,
            linestyle="--"
        )

        plt.title(
            f"{self.model_type.upper()} "
            f"— Train"
        )

        plt.xlabel(
            train_xlabel
        )

        plt.ylabel(
            self.label_type
            if self.label_type
            is not None
            else "Target"
        )

        plt.legend()

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        plt.show()

        oof_df = (
            self.oof_predictions
            .copy()
        )

        if (
            "cutoff_date"
            in oof_df.columns
        ):

            oof_df[
                "cutoff_date"
            ] = (
                pd.to_datetime(
                    oof_df[
                        "cutoff_date"
                    ]
                )
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

            x_oof = (
                oof_df[
                    "cutoff_date"
                ]
            )

            oof_xlabel = "Date"

        else:

            if (
                "row_index"
                in oof_df.columns
            ):

                oof_df = (
                    oof_df
                    .sort_values(
                        "row_index"
                    )
                    .reset_index(
                        drop=True
                    )
                )

            x_oof = (
                np.arange(
                    len(oof_df)
                )
            )

            oof_xlabel = (
                "Observation"
            )

        plt.figure(
            figsize=(12, 5)
        )

        plt.plot(
            x_oof,
            oof_df[
                "y_true"
            ],
            label="Actual"
        )

        plt.plot(
            x_oof,
            oof_df[
                "y_pred"
            ],
            label="OOF prediction"
        )

        plt.axhline(
            0,
            linewidth=0.8,
            linestyle="--"
        )

        plt.title(
            f"{self.model_type.upper()} "
            f"— Out-of-Fold Train"
        )

        plt.xlabel(
            oof_xlabel
        )

        plt.ylabel(
            self.label_type
            if self.label_type
            is not None
            else "Target"
        )

        plt.legend()

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        plt.show()

        plt.figure(
            figsize=(12, 5)
        )

        plt.plot(
            x_test,
            np.asarray(
                self.y_test
            ),
            label="Actual"
        )

        plt.plot(
            x_test,
            test_predictions,
            label="Prediction"
        )

        plt.axhline(
            0,
            linewidth=0.8,
            linestyle="--"
        )

        plt.title(
            f"{self.model_type.upper()} "
            f"— Test"
        )

        plt.xlabel(
            test_xlabel
        )

        plt.ylabel(
            self.label_type
            if self.label_type
            is not None
            else "Target"
        )

        plt.legend()

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        plt.show()