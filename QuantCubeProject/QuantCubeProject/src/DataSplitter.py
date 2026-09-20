class DataSplitter:

    def __init__(
        self,
        df,
        target_col="target",
        date_col="cutoff_date",
        test_size=0.1,
    ):

        self.df = (
            df
            .sort_values(date_col)
            .reset_index(drop=True)
        )

        self.target_col = target_col
        self.date_col = date_col
        self.test_size = test_size


    def train_test_split(self):


        excluded_cols = [
            self.target_col,
            self.date_col,
            "gdp_initial",
            "last_quarter_gdp",
            "label_type",
            "initial_growth",
            "cutoff_date",
            "quarter_end"
        ]

        feature_cols = [
            col for col in self.df.columns
            if col not in excluded_cols
        ]

        X = self.df[feature_cols]

        y = self.df[
            self.target_col
        ]

        dates = self.df[
            self.date_col
        ]


        valid_mask = y.notna()

        X = (
            X.loc[valid_mask]
            .reset_index(drop=True)
        )

        y = (
            y.loc[valid_mask]
            .reset_index(drop=True)
        )

        dates = (
            dates.loc[valid_mask]
            .reset_index(drop=True)
        )


        split_idx = int(
            len(X)
            * (1 - self.test_size)
        )



        X_train = X.iloc[
            :split_idx
        ]

        y_train = y.iloc[
            :split_idx
        ]

        dates_train = dates.iloc[
            :split_idx
        ]


        X_test = X.iloc[
            split_idx:
        ]

        y_test = y.iloc[
            split_idx:
        ]

        dates_test = dates.iloc[
            split_idx:
        ]

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            dates_train,
            dates_test
        )
