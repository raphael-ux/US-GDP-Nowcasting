import time
import requests
import pandas as pd

from pathlib import Path
from io import BytesIO

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class DataCollection:
    """
    Collect point-in-time macroeconomic data from the FRED API.

    Main functionalities
    --------------------
    - Retrieve a FRED series.
    - Retrieve initial GDP release dates.
    - Build the point-in-time GDP growth target.
    - Construct cutoff dates for the nowcasting exercise.
    - Retrieve explanatory variables as known at each cutoff.
    - Retrieve FRED metadata.
    - Retrieve SF Fed Daily News Sentiment data.
    """

    BASE_URL = "https://api.stlouisfed.org/fred"


    def __init__(
        self,
        api_key_fed,
        timeout=30,
        max_retries=5,
        backoff_factor=1
    ):

        self.api_key_fed = api_key_fed
        self.timeout = timeout

        self.project_root = (
            Path(__file__)
            .resolve()
            .parents[1]
        )

        self.session = requests.Session()

        retry_strategy = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            status=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[
                429,
                500,
                502,
                503,
                504
            ],
            allowed_methods=["GET"]
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy
        )

        self.session.mount(
            "https://",
            adapter
        )

        self.calendar = None

    def _request(
        self,
        endpoint,
        params
    ):
        """
        Send a GET request to FRED and return JSON response.
        """

        url = (
            f"{self.BASE_URL}/{endpoint}"
        )

        params = {
            **params,
            "api_key": self.api_key_fed,
            "file_type": "json"
        }

        try:

            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout
            )

            response.raise_for_status()

            data = response.json()

        except requests.exceptions.Timeout as exc:

            raise RuntimeError(
                f"FRED request timed out "
                f"for {endpoint}"
            ) from exc

        except requests.exceptions.ConnectionError as exc:

            raise RuntimeError(
                f"Connection error while "
                f"requesting {endpoint}"
            ) from exc

        except requests.exceptions.HTTPError as exc:

            # Keep FRED error message if available
            try:
                fred_message = response.text
            except Exception:
                fred_message = ""

            raise RuntimeError(
                f"FRED HTTP error: {exc}\n"
                f"{fred_message}"
            ) from exc

        except ValueError as exc:

            raise RuntimeError(
                "FRED returned an invalid "
                "JSON response."
            ) from exc

        return data

    def get_series(
        self,
        series_id,
        output_type=1,
        realtime_start=None,
        realtime_end=None,
        observation_start=None,
        observation_end=None,
        vintage_dates=None
    ):
        """
        Retrieve observations for one FRED series.
        """

        params = {
            "series_id": series_id,
            "output_type": output_type
        }

        if realtime_start is not None:
            params["realtime_start"] = (
                realtime_start
            )

        if realtime_end is not None:
            params["realtime_end"] = (
                realtime_end
            )

        if observation_start is not None:
            params["observation_start"] = (
                observation_start
            )

        if observation_end is not None:
            params["observation_end"] = (
                observation_end
            )

        if vintage_dates is not None:

            if isinstance(
                vintage_dates,
                (list, tuple)
            ):
                vintage_dates = ",".join(
                    vintage_dates
                )

            params["vintage_dates"] = (
                vintage_dates
            )

        return self._request(
            endpoint="series/observations",
            params=params
        )


    @staticmethod
    def _format_initial_release(
        data,
        series_id=None,
        feature_name=None
    ):
        """
        Convert FRED initial-release observations
        into a clean DataFrame.
        """

        observations = data.get(
            "observations",
            []
        )

        if not observations:

            return pd.DataFrame(
                columns=[
                    "observation_date",
                    "availability_date",
                    "initial_value",
                    "series_id",
                    "feature_name"
                ]
            )

        df = pd.DataFrame(
            observations
        )

        required_columns = {
            "date",
            "realtime_start",
            "value"
        }

        missing_columns = (
            required_columns
            - set(df.columns)
        )

        if missing_columns:

            raise ValueError(
                f"Missing FRED fields: "
                f"{missing_columns}"
            )

        df = df[
            [
                "date",
                "realtime_start",
                "value"
            ]
        ].copy()

        df = df.rename(
            columns={
                "date":
                    "observation_date",

                "realtime_start":
                    "availability_date",

                "value":
                    "initial_value"
            }
        )

        df["observation_date"] = (
            pd.to_datetime(
                df["observation_date"],
                errors="coerce"
            )
        )

        df["availability_date"] = (
            pd.to_datetime(
                df["availability_date"],
                errors="coerce"
            )
        )

        df["initial_value"] = (
            pd.to_numeric(
                df["initial_value"],
                errors="coerce"
            )
        )

        if series_id is not None:

            df["series_id"] = (
                series_id
            )

        if feature_name is not None:

            df["feature_name"] = (
                feature_name
            )

        df = df.dropna(
            subset=[
                "observation_date",
                "availability_date"
            ]
        )

        df = (
            df
            .sort_values(
                [
                    "observation_date",
                    "availability_date"
                ]
            )
            .reset_index(drop=True)
        )

        return df

    def get_release_dates(
        self,
        series_id="GDPC1",
        observation_start=None
    ):
        """
        Retrieve the initial release date of GDP
        for every quarter.

        Important
        ---------
        This function does NOT define the model cutoff.

        The cutoff belongs to the nowcasting experiment
        and is constructed later in
        build_initial_gdp_growth().
        """

        data = self.get_series(
            series_id=series_id,
            output_type=4,

            realtime_start="1776-07-04",
            realtime_end="9999-12-31",

            observation_start=(
                observation_start
            )
        )

        df = self._format_initial_release(
            data=data,
            series_id=series_id
        )

        df = df.rename(
            columns={
                "observation_date":
                    "quarter_start",

                "availability_date":
                    "release_date"
            }
        )


        df["quarter_end"] = (
            df["quarter_start"]
            .dt.to_period("Q")
            .dt.end_time
            .dt.normalize()
        )

        df["quarter"] = (
            df["quarter_start"]
            .dt.to_period("Q")
            .astype(str)
        )

        df["release_lag_days"] = (
            df["release_date"]
            - df["quarter_end"]
        ).dt.days

        df = df[
            [
                "quarter",
                "quarter_start",
                "quarter_end",
                "release_date",
                "release_lag_days",
                "initial_value"
            ]
        ].reset_index(drop=True)

        raw_data_dir = (
            self.project_root
            / "data"
            / "raw"
        )

        raw_data_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            raw_data_dir
            / "calendar_gdp_release_dates.csv"
        )

        df.to_csv(
            output_path,
            index=False
        )

        print(
            f"GDP release calendar saved to: "
            f"{output_path}"
        )

        self.calendar = df

        return df

    def get_raw_features(
        self,
        fred_features,
        observation_start=None,
        delay=0.25
    ):
        """
        Download initial releases for explanatory variables.

        Parameters
        ----------
        fred_features : dict
            Example:
            {
                "INDPRO":
                    "industrial_production",
                ...
            }

        observation_start : str, optional
            Earliest observation date.

        delay : float
            Delay between API requests.
        """

        all_features = []
        failed_series = []

        for (
            series_id,
            feature_name
        ) in fred_features.items():

            print(
                f"Downloading "
                f"{series_id:<12} "
                f"- {feature_name}"
            )

            try:

                data = self.get_series(
                    series_id=series_id,

                    # Initial releases
                    output_type=4,

                    realtime_start=(
                        "1776-07-04"
                    ),

                    realtime_end=(
                        "9999-12-31"
                    ),

                    observation_start=(
                        observation_start
                    )
                )

                df = (
                    self
                    ._format_initial_release(
                        data=data,
                        series_id=series_id,
                        feature_name=(
                            feature_name
                        )
                    )
                )

                if df.empty:

                    print(
                        "Warning: no data "
                        f"returned for "
                        f"{series_id}"
                    )

                    failed_series.append(
                        series_id
                    )

                else:

                    all_features.append(
                        df
                    )

            except Exception as exc:

                print(
                    f"Failed {series_id}: "
                    f"{exc}"
                )

                failed_series.append(
                    series_id
                )

            time.sleep(delay)

        if not all_features:

            raise RuntimeError(
                "No FRED feature could "
                "be downloaded."
            )

        raw_features = pd.concat(
            all_features,
            ignore_index=True
        )

        raw_features = (
            raw_features
            .drop_duplicates(
                subset=[
                    "series_id",
                    "observation_date",
                    "availability_date"
                ]
            )
            .sort_values(
                [
                    "series_id",
                    "observation_date"
                ]
            )
            .reset_index(drop=True)
        )

        if failed_series:

            print(
                "\nFailed series:",
                ", ".join(
                    failed_series
                )
            )

        return raw_features


    def get_features_by_cutoff(
        self,
        calendar,
        fred_features,
        cutoff_col="cutoff_date",
        lookback_quarters=3,
        delay=0.1
    ):
        """
        Construct FRED feature snapshots for each
        nowcasting cutoff.

        Important
        ---------
        cutoff_date:
            determines WHAT INFORMATION WAS KNOWN.

        quarter_end:
            determines WHICH ECONOMIC PERIOD
            observations may belong to.

        Therefore:

            realtime = cutoff_date

        while:

            observation_end = quarter_end
        """

        snapshots = {}

        calendar = (
            calendar
            .copy()
        )

        calendar[cutoff_col] = (
            pd.to_datetime(
                calendar[cutoff_col]
            )
        )

        calendar["quarter_end"] = (
            pd.to_datetime(
                calendar["quarter_end"]
            )
        )

        if calendar[
            cutoff_col
        ].isna().any():

            raise ValueError(
                "Missing cutoff dates "
                "in calendar."
            )

        if calendar[
            "quarter_end"
        ].isna().any():

            raise ValueError(
                "Missing quarter-end dates "
                "in calendar."
            )

        cutoff_dates = (
            calendar[cutoff_col]
            .drop_duplicates()
            .sort_values()
        )
        
        for cutoff_date in cutoff_dates:

            calendar_now = (
                calendar[
                    calendar[
                        cutoff_col
                    ] == cutoff_date
                ]
            )

            if len(calendar_now) != 1:

                raise ValueError(
                    "Each cutoff date must "
                    "correspond to exactly "
                    "one GDP quarter. "
                    f"Problem for "
                    f"{cutoff_date}."
                )

            quarter_end = (
                pd.Timestamp(
                    calendar_now[
                        "quarter_end"
                    ].iloc[0]
                )
            )

            target_quarter = (
                quarter_end
                .to_period("Q")
            )

            print(
                f"\nBuilding snapshot for "
                f"{target_quarter} "
                f"| cutoff "
                f"{cutoff_date.date()}"
            )
            
            first_quarter = (
                target_quarter
                - (
                    lookback_quarters
                    - 1
                )
            )

            observation_start = (
                first_quarter
                .start_time
            )

            observation_end = (
                quarter_end
            )


            cutoff_str = (
                cutoff_date.strftime(
                    "%Y-%m-%d"
                )
            )

            observation_start_str = (
                observation_start.strftime(
                    "%Y-%m-%d"
                )
            )

            observation_end_str = (
                observation_end.strftime(
                    "%Y-%m-%d"
                )
            )

            snapshot_features = []


            for (
                series_id,
                feature_name
            ) in fred_features.items():

                print(
                    f"  {series_id:<12} "
                    f"- {feature_name}"
                )

                try:

                    data = self.get_series(
                        series_id=series_id,
                        output_type=1,

                        realtime_start=(
                            cutoff_str
                        ),

                        realtime_end=(
                            cutoff_str
                        ),


                        observation_start=(
                            observation_start_str
                        ),

                        observation_end=(
                            observation_end_str
                        )
                    )

                    observations = (
                        data.get(
                            "observations",
                            []
                        )
                    )

                    if not observations:
                        continue

                    df = pd.DataFrame(
                        observations
                    )

                    required_columns = [
                        "date",
                        "realtime_start",
                        "realtime_end",
                        "value"
                    ]

                    missing = (
                        set(required_columns)
                        - set(df.columns)
                    )

                    if missing:

                        raise ValueError(
                            f"Missing FRED "
                            f"columns: "
                            f"{missing}"
                        )

                    df = df[
                        required_columns
                    ].copy()

                    df = df.rename(
                        columns={
                            "date":
                                "observation_date"
                        }
                    )

                    df[
                        "observation_date"
                    ] = pd.to_datetime(
                        df[
                            "observation_date"
                        ],
                        errors="coerce"
                    )

                    df[
                        "realtime_start"
                    ] = pd.to_datetime(
                        df[
                            "realtime_start"
                        ],
                        errors="coerce"
                    )

                    df[
                        "realtime_end"
                    ] = pd.to_datetime(
                        df[
                            "realtime_end"
                        ],
                        errors="coerce"
                    )

                    df["value"] = (
                        pd.to_numeric(
                            df["value"],
                            errors="coerce"
                        )
                    )


                    df[
                        "cutoff_date"
                    ] = cutoff_date

                    df[
                        "quarter_end"
                    ] = quarter_end

                    df[
                        "series_id"
                    ] = series_id

                    df[
                        "feature_name"
                    ] = feature_name

                    # Defensive check:
                    # no future-quarter references
                    df = df[
                        df[
                            "observation_date"
                        ]
                        <= quarter_end
                    ]

                    snapshot_features.append(
                        df
                    )

                except Exception as exc:

                    print(
                        f"  Failed "
                        f"{series_id}: "
                        f"{exc}"
                    )

                time.sleep(delay)

            if snapshot_features:

                snapshot = pd.concat(
                    snapshot_features,
                    ignore_index=True
                )

                snapshot = (
                    snapshot
                    .sort_values(
                        [
                            "series_id",
                            "observation_date"
                        ]
                    )
                    .reset_index(
                        drop=True
                    )
                )

                snapshots[
                    cutoff_date
                ] = snapshot


        raw_data_dir = (
            self.project_root
            / "data"
            / "raw"
        )

        raw_data_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            raw_data_dir
            / "raw_features.pkl"
        )

        pd.to_pickle(
            snapshots,
            output_path
        )

        print(
            f"\nRaw features saved to: "
            f"{output_path}"
        )

        return snapshots

    def get_series_metadata(
        self,
        series_id
    ):
        """
        Retrieve metadata for a FRED series.
        """

        data = self._request(
            endpoint="series",
            params={
                "series_id":
                    series_id
            }
        )

        if (
            "seriess" not in data
            or len(
                data["seriess"]
            ) == 0
        ):

            raise ValueError(
                f"No metadata found "
                f"for {series_id}"
            )

        return data[
            "seriess"
        ][0]


    def build_preprocessing_config(
        self,
        fred_features
    ):
        """
        Build preprocessing metadata
        for each FRED feature.
        """

        config = {}

        for (
            series_id,
            feature_name
        ) in fred_features.items():

            metadata = (
                self
                .get_series_metadata(
                    series_id
                )
            )

            config[
                series_id
            ] = {

                "feature_name":
                    feature_name,

                "frequency":
                    metadata[
                        "frequency_short"
                    ],

                "units":
                    metadata[
                        "units"
                    ],

                "seasonal_adjustment":
                    metadata[
                        "seasonal_adjustment_short"
                    ]
            }

        return config
    
    def build_normalized_gdp_level(
        self,
        growth_df,
        base_value=1.0
    ):
        """
        Construct an artificial normalized GDP index
        from point-in-time annualized quarterly growth.

        This index is used only as a reconstructed
        continuous level.

        It is NOT the actual historical GDP level.
        """

        df = (
            growth_df
            .sort_values(
                "quarter_start"
            )
            .reset_index(drop=True)
            .copy()
        )


        df["quarterly_ratio"] = (
            1
            + df[
                "initial_growth"
            ] / 100
        ) ** (1 / 4)


        normalized_level = [
            base_value
        ]

        for i in range(
            1,
            len(df)
        ):

            new_level = (
                normalized_level[-1]
                * df.loc[
                    i,
                    "quarterly_ratio"
                ]
            )

            normalized_level.append(
                new_level
            )

        df["initial_value"] = (
            normalized_level
        )

        return df[
            [
                "quarter",
                "quarter_start",
                "quarter_end",
                "release_date",
                "release_lag_days",
                "initial_value",
                "initial_growth",
                "quarterly_ratio",
                "cutoff_date"
            ]
        ]

    def build_initial_gdp_growth(
        self,
        cutoff_days=18
    ):
        """
        Construct point-in-time annualized GDP growth.

        Parameters
        ----------
        cutoff_days : int
            Number of calendar days after quarter-end
            at which the nowcast is produced.

            Examples:
                0
                6
                12
                18

        Method
        ------
        For every quarter:

        1. Take the initial GDP release date.
        2. Retrieve GDP_t and GDP_{t-1}
           from the SAME release-date vintage.
        3. Calculate annualized q/q growth.
        4. Define:

           cutoff_date
           =
           quarter_end + cutoff_days

        The cutoff must always occur before
        the initial GDP release.
        """

        if self.calendar is None:

            raise ValueError(
                "GDP release calendar has not "
                "been created. Call "
                "get_release_dates() first."
            )

        if cutoff_days < 0:

            raise ValueError(
                "cutoff_days must be >= 0."
            )

        rows = []

        for _, row in (
            self.calendar.iterrows()
        ):


            quarter_start = (
                pd.Timestamp(
                    row[
                        "quarter_start"
                    ]
                )
            )

            quarter_end = (
                pd.Timestamp(
                    row[
                        "quarter_end"
                    ]
                )
            )

            release_date = (
                pd.Timestamp(
                    row[
                        "release_date"
                    ]
                )
            )

            print(
                "Get GDP values for "
                f"{row['quarter']} "
                f"| release "
                f"{release_date.date()}"
            )


            cutoff_date = (
                quarter_end
                + pd.Timedelta(
                    days=cutoff_days
                )
            )


            if cutoff_date >= release_date:

                raise ValueError(
                    f"Invalid cutoff for "
                    f"{row['quarter']}: "
                    f"cutoff "
                    f"{cutoff_date.date()} "
                    f"is not before GDP "
                    f"release "
                    f"{release_date.date()}."
                )


            previous_quarter_start = (
                quarter_start
                - pd.offsets.QuarterBegin(
                    startingMonth=1
                )
            )
            data = self.get_series(
                series_id="GDPC1",

                realtime_start=(
                    release_date
                    .strftime(
                        "%Y-%m-%d"
                    )
                ),

                realtime_end=(
                    release_date
                    .strftime(
                        "%Y-%m-%d"
                    )
                ),

                observation_start=(
                    previous_quarter_start
                    .strftime(
                        "%Y-%m-%d"
                    )
                ),

                observation_end=(
                    quarter_start
                    .strftime(
                        "%Y-%m-%d"
                    )
                ),

                output_type=1
            )

            observations = (
                data.get(
                    "observations",
                    []
                )
            )

            if not observations:

                print(
                    f"Skipping "
                    f"{row['quarter']}: "
                    "no GDP observations."
                )

                continue

            obs = pd.DataFrame(
                observations
            )

            obs["date"] = (
                pd.to_datetime(
                    obs["date"],
                    errors="coerce"
                )
            )

            obs["value"] = (
                pd.to_numeric(
                    obs["value"],
                    errors="coerce"
                )
            )

            obs = (
                obs
                .dropna(
                    subset=[
                        "date",
                        "value"
                    ]
                )
                .sort_values(
                    "date"
                )
                .reset_index(
                    drop=True
                )
            )


            if len(obs) < 2:

                print(
                    f"Skipping "
                    f"{row['quarter']}: "
                    "fewer than two GDP "
                    "observations."
                )

                continue

            previous_gdp = (
                obs.iloc[-2][
                    "value"
                ]
            )

            current_gdp = (
                obs.iloc[-1][
                    "value"
                ]
            )


            initial_growth = (
                100
                * (
                    (
                        current_gdp
                        / previous_gdp
                    ) ** 4
                    - 1
                )
            )

            rows.append(
                {
                    "quarter":
                        row["quarter"],

                    "quarter_start":
                        quarter_start,

                    "quarter_end":
                        quarter_end,

                    "release_date":
                        release_date,

                    "release_lag_days":
                        row[
                            "release_lag_days"
                        ],

                    "initial_growth":
                        initial_growth,

                    "cutoff_date":
                        cutoff_date
                }
            )

        if not rows:

            raise RuntimeError(
                "No GDP growth observations "
                "could be constructed."
            )

        grow_df = pd.DataFrame(
            rows
        )


        if not (
            grow_df[
                "cutoff_date"
            ]
            <
            grow_df[
                "release_date"
            ]
        ).all():

            invalid = grow_df[
                grow_df[
                    "cutoff_date"
                ]
                >=
                grow_df[
                    "release_date"
                ]
            ]

            raise ValueError(
                "Some cutoff dates occur "
                "on or after the GDP release:\n"
                f"{invalid}"
            )


        grow_df_normalized = (
            self
            .build_normalized_gdp_level(
                grow_df
            )
        )



        raw_data_dir = (
            self.project_root
            / "data"
            / "raw"
        )

        raw_data_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            raw_data_dir
            / "calendar_gdp.csv"
        )

        grow_df_normalized.to_csv(
            output_path,
            index=False
        )

        print(
            f"GDP calendar saved to: "
            f"{output_path}"
        )

        print(
            f"Cutoff horizon: "
            f"{cutoff_days} days "
            f"after quarter-end"
        )

        self.calendar = (
            grow_df_normalized
        )

        return grow_df_normalized


    def get_sf_fed_news_sentiment(
        self,
        save=True
    ):
        """
        Download the San Francisco Fed
        Daily News Sentiment Index.
        """

        url = (
            "https://www.frbsf.org/"
            "wp-content/uploads/"
            "news_sentiment_data.xlsx"
        )

        print(
            "Downloading SF Fed "
            "Daily News Sentiment..."
        )

        response = (
            self.session.get(
                url,
                timeout=self.timeout
            )
        )

        response.raise_for_status()


        df = pd.read_excel(
            BytesIO(
                response.content
            ),
            sheet_name="Data"
        )


        df = df.rename(
            columns={
                "date":
                    "observation_date",

                "News Sentiment":
                    "news_sentiment"
            }
        )

        df[
            "observation_date"
        ] = pd.to_datetime(
            df[
                "observation_date"
            ],
            errors="coerce"
        )

        df[
            "news_sentiment"
        ] = pd.to_numeric(
            df[
                "news_sentiment"
            ],
            errors="coerce"
        )

        df = (
            df
            .dropna(
                subset=[
                    "observation_date",
                    "news_sentiment"
                ]
            )
            .sort_values(
                "observation_date"
            )
            .reset_index(
                drop=True
            )
        )



        if save:

            raw_data_dir = (
                self.project_root
                / "data"
                / "raw"
                / "alternative"
            )

            raw_data_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            # Original Excel file
            excel_path = (
                raw_data_dir
                / "sf_fed_news_sentiment.xlsx"
            )

            with open(
                excel_path,
                "wb"
            ) as f:

                f.write(
                    response.content
                )

            # Clean CSV
            csv_path = (
                raw_data_dir
                / "sf_fed_news_sentiment.csv"
            )

            df.to_csv(
                csv_path,
                index=False
            )

            print(
                "News sentiment saved to: "
                f"{csv_path}"
            )

        return df