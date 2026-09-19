import time
import requests
import pandas as pd
from pathlib import Path

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class DataCollection:
    """
    Collect point-in-time macroeconomic data from the FRED API.

    Main functionalities
    --------------------
    - Retrieve a FRED series.
    - Retrieve initial GDP releases and their release dates.
    - Retrieve initial releases of explanatory features.
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
        self.project_root = Path(__file__).resolve().parents[1]

        # Persistent HTTP session
        self.session = requests.Session()

        # Retry strategy for temporary connection/server problems
        retry_strategy = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            status=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[
                429,  # Too many requests
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

    # ---------------------------------------------------------
    # Internal API request
    # ---------------------------------------------------------

    def _request(self, endpoint, params):
        """
        Send a GET request to FRED and return the JSON response.
        """

        url = f"{self.BASE_URL}/{endpoint}"

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
                f"FRED request timed out for {endpoint}"
            ) from exc

        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Connection error while requesting {endpoint}"
            ) from exc

        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(
                f"FRED HTTP error: {exc}"
            ) from exc

        except ValueError as exc:
            raise RuntimeError(
                "FRED returned an invalid JSON response."
            ) from exc

        return data

    # ---------------------------------------------------------
    # Generic series retrieval
    # ---------------------------------------------------------

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

        params = {
            "series_id": series_id,
            "output_type": output_type
        }

        if realtime_start is not None:
            params["realtime_start"] = realtime_start

        if realtime_end is not None:
            params["realtime_end"] = realtime_end

        if observation_start is not None:
            params["observation_start"] = observation_start

        if observation_end is not None:
            params["observation_end"] = observation_end

        if vintage_dates is not None:

            if isinstance(vintage_dates, (list, tuple)):
                vintage_dates = ",".join(vintage_dates)

            params["vintage_dates"] = vintage_dates

        return self._request(
            endpoint="series/observations",
            params=params
        )

    # ---------------------------------------------------------
    # Internal observation formatter
    # ---------------------------------------------------------

    @staticmethod
    def _format_initial_release(
        data,
        series_id=None,
        feature_name=None
    ):
        """
        Convert FRED initial-release observations to a clean DataFrame.
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

        df = pd.DataFrame(observations)

        required_columns = {
            "date",
            "realtime_start",
            "value"
        }

        missing_columns = (
            required_columns - set(df.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing FRED fields: {missing_columns}"
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
                "date": "observation_date",
                "realtime_start": "availability_date",
                "value": "initial_value"
            }
        )

        df["observation_date"] = pd.to_datetime(
            df["observation_date"],
            errors="coerce"
        )

        df["availability_date"] = pd.to_datetime(
            df["availability_date"],
            errors="coerce"
        )

        df["initial_value"] = pd.to_numeric(
            df["initial_value"],
            errors="coerce"
        )

        if series_id is not None:
            df["series_id"] = series_id

        if feature_name is not None:
            df["feature_name"] = feature_name

        # Remove observations with invalid dates
        df = df.dropna(
            subset=[
                "observation_date",
                "availability_date"
            ]
        )

        # Ensure deterministic ordering
        df = df.sort_values(
            [
                "observation_date",
                "availability_date"
            ]
        ).reset_index(drop=True)

        return df

    # ---------------------------------------------------------
    # GDP initial releases
    # ---------------------------------------------------------

    def get_release_dates(
        self,
        series_id="GDPC1",
        observation_start=None
    ):
        """
        Retrieve the initial release of GDP for each quarter
        and save it in data/raw/calendar_gdp.csv.
        """

        data = self.get_series(
            series_id=series_id,
            output_type=4,
            realtime_start="1776-07-04",
            realtime_end="9999-12-31",
            observation_start=observation_start
        )

        df = self._format_initial_release(
            data=data,
            series_id=series_id
        )

        df = df.rename(
            columns={
                "observation_date": "quarter_start",
                "availability_date": "release_date"
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
        
        df["cutoff_date"] = df["quarter_end"] + pd.Timedelta(days=0) #try 0, 6, 12, 18

        df = df[
            [
                "quarter",
                "quarter_start",
                "quarter_end",
                "release_date",
                "release_lag_days",
                "initial_value",
                "cutoff_date"
            ]
        ].reset_index(drop=True)

        # --------------------------------------------------
        # Save raw GDP calendar
        # --------------------------------------------------


        raw_data_dir = self.project_root / "data" / "raw"

        raw_data_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = raw_data_dir / "calendar_gdp.csv"

        df.to_csv(
            output_path,
            index=False
        )

        print(f"GDP calendar saved to: {output_path}")
        
        self.calendar = df

        return df

    # ---------------------------------------------------------
    # Raw explanatory features
    # ---------------------------------------------------------

    def get_raw_features(
        self,
        fred_features,
        observation_start=None,
        delay=0.25
    ):
        """
        Download initial releases for all explanatory variables.

        Parameters
        ----------
        fred_features : dict
            Mapping:
            {
                "INDPRO": "industrial_production",
                ...
            }

        observation_start : str, optional
            Earliest observation date to retrieve.

        delay : float
            Pause between API calls to avoid sending requests
            too aggressively.

        Returns
        -------
        pd.DataFrame
            Long-format point-in-time feature dataset.
        """

        all_features = []
        failed_series = []

        for series_id, feature_name in fred_features.items():

            print(
                f"Downloading {series_id:<12} "
                f"- {feature_name}"
            )

            try:

                data = self.get_series(
                    series_id=series_id,
                    output_type=4, # to get point in time information
                    realtime_start="1776-07-04",
                    realtime_end="9999-12-31",
                    observation_start=observation_start
                )

                df = self._format_initial_release(
                    data=data,
                    series_id=series_id,
                    feature_name=feature_name
                )

                if df.empty:
                    print(
                        f"Warning: no data returned for "
                        f"{series_id}"
                    )
                    failed_series.append(series_id)

                else:
                    all_features.append(df)

            except Exception as exc:

                print(
                    f"Failed {series_id}: {exc}"
                )

                failed_series.append(series_id)

            time.sleep(delay)

        if not all_features:
            raise RuntimeError(
                "No FRED feature could be downloaded."
            )

        raw_features = pd.concat(
            all_features,
            ignore_index=True
        )

        # Remove exact duplicates if FRED ever returns any
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
                ", ".join(failed_series)
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

        snapshots = {}

        # --------------------------------------------------
        # Prepare calendar dates
        # --------------------------------------------------

        calendar = calendar.copy()

        calendar[cutoff_col] = pd.to_datetime(
            calendar[cutoff_col]
        )

        calendar["quarter_end"] = pd.to_datetime(
            calendar["quarter_end"]
        )

        cutoff_dates = (
            calendar[cutoff_col]
            .drop_duplicates()
            .sort_values()
        )

        for cutoff_date in cutoff_dates:

            # --------------------------------------------------
            # Get calendar row associated with this cutoff
            # --------------------------------------------------

            calendar_now = (
                calendar[
                    calendar[cutoff_col] == cutoff_date
                ]
            )

            quarter_end = pd.Timestamp(
                calendar_now[
                    "quarter_end"
                ].iloc[0]
            )

            print(
                f"\nBuilding snapshot for "
                f"{quarter_end.to_period('Q')} "
                f"| cutoff {cutoff_date.date()}"
            )

            # --------------------------------------------------
            # Historical observation window
            # --------------------------------------------------

            current_quarter = (
                quarter_end.to_period("Q")
            )

            first_quarter = (
                current_quarter
                - (lookback_quarters - 1)
            )

            observation_start = (
                first_quarter.start_time
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
                quarter_end.strftime(
                    "%Y-%m-%d"
                )
            )

            snapshot_features = []

            # --------------------------------------------------
            # Download every feature as known at cutoff
            # --------------------------------------------------

            for series_id, feature_name in fred_features.items():

                print(
                    f"  {series_id:<12} - {feature_name}"
                )

                try:

                    data = self.get_series(
                        series_id=series_id,
                        output_type=1,

                        # Vintage known at cutoff
                        realtime_start=cutoff_str,
                        realtime_end=cutoff_str,

                        # Economic observations only through
                        # the end of the target GDP quarter
                        observation_start=observation_start_str,
                        observation_end=observation_end_str
                    )

                    df = pd.DataFrame(
                        data["observations"]
                    )

                    if df.empty:
                        continue

                    df = df[
                        [
                            "date",
                            "realtime_start",
                            "realtime_end",
                            "value"
                        ]
                    ].copy()

                    df = df.rename(
                        columns={
                            "date": "observation_date"
                        }
                    )

                    df["observation_date"] = pd.to_datetime(
                        df["observation_date"]
                    )

                    df["realtime_start"] = pd.to_datetime(
                        df["realtime_start"]
                    )

                    df["realtime_end"] = pd.to_datetime(
                        df["realtime_end"]
                    )

                    df["value"] = pd.to_numeric(
                        df["value"],
                        errors="coerce"
                    )

                    df["cutoff_date"] = cutoff_date
                    df["quarter_end"] = quarter_end
                    df["series_id"] = series_id
                    df["feature_name"] = feature_name

                    snapshot_features.append(
                        df
                    )

                except Exception as exc:

                    print(
                        f"  Failed {series_id}: {exc}"
                    )

                time.sleep(delay)

            # --------------------------------------------------
            # One DataFrame for this cutoff
            # --------------------------------------------------

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

        # --------------------------------------------------
        # Save snapshots
        # --------------------------------------------------

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
    
    
    def get_series_metadata(self, series_id):
        '''
        get the frequence of updates for each features
        '''
        

        data = self._request(
            endpoint="series",
            params={
                "series_id": series_id
            }
        )

        if "seriess" not in data or len(data["seriess"]) == 0:
            raise ValueError(
                f"No metadata found for {series_id}"
            )

        return data["seriess"][0]
    
    def build_preprocessing_config(
        self,
        fred_features
    ):

        config = {}

        for series_id, feature_name in fred_features.items():

            metadata = self.get_series_metadata(
                series_id
            )

            config[series_id] = {
                "feature_name": feature_name,
                "frequency": metadata["frequency_short"],
                "units": metadata["units"],
                "seasonal_adjustment": metadata[
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
        Build a continuous normalized GDP level from
        same-vintage annualized quarterly GDP growth.

        Parameters
        ----------
        growth_df : pd.DataFrame
            Must contain:
                - quarter
                - quarter_start
                - quarter_end
                - release_date
                - release_lag_days
                - initial_growth

        base_value : float
            Arbitrary starting level.
            1.0 or 100.0 are natural choices.

        Returns
        -------
        pd.DataFrame
        """

        df = (
            growth_df
            .sort_values("quarter_start")
            .reset_index(drop=True)
            .copy()
        )

        # --------------------------------------------------
        # Convert annualized growth back to quarterly ratio
        # --------------------------------------------------

        df["quarterly_ratio"] = (
            1
            + df["initial_growth"] / 100
        ) ** (1 / 4)

        # --------------------------------------------------
        # Construct artificial continuous GDP level
        # --------------------------------------------------

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
    
    
    def build_initial_gdp_growth(self):

        rows = []

        for _, row in self.calendar.iterrows():

            quarter_start = pd.Timestamp(
                row["quarter_start"]
            )
            
            print(f'get value for quarter_start : {quarter_start}')

            release_date = pd.Timestamp(
                row["release_date"]
            )

            previous_quarter_start = (
                quarter_start
                - pd.offsets.QuarterBegin(startingMonth=1)
            )

            # -----------------------------------------
            # Get GDP values from SAME vintage
            # -----------------------------------------

            data = self.get_series(
                series_id="GDPC1",
                realtime_start=release_date.strftime("%Y-%m-%d"),
                realtime_end=release_date.strftime("%Y-%m-%d"),
                observation_start=previous_quarter_start.strftime("%Y-%m-%d"),
                observation_end=quarter_start.strftime("%Y-%m-%d"),
                output_type=1
            )

            obs = pd.DataFrame(
                data["observations"]
            )

            obs["date"] = pd.to_datetime(
                obs["date"]
            )

            obs["value"] = pd.to_numeric(
                obs["value"],
                errors="coerce"
            )

            obs = (
                obs
                .dropna(subset=["value"])
                .sort_values("date")
            )

            if len(obs) < 2:
                continue

            previous_gdp = obs.iloc[-2]["value"]
            current_gdp = obs.iloc[-1]["value"]

            # -----------------------------------------
            # Annualized quarter-over-quarter growth
            # -----------------------------------------

            initial_growth = (
                100
                * (
                    (current_gdp / previous_gdp) ** 4
                    - 1
                )
            ) 
            

            rows.append({
                "quarter": row["quarter"],
                "quarter_start": quarter_start,
                "quarter_end": row["quarter_end"],
                "release_date": release_date,
                "release_lag_days": row["release_lag_days"],
                "initial_growth": initial_growth,
                "cutoff_date":row["quarter_end"] + pd.Timedelta(days=18)
            })
            
        grow_df = pd.DataFrame(rows)
            
        grow_df_normalized = self.build_normalized_gdp_level(grow_df)
            
        raw_data_dir = self.project_root / "data" / "raw"
            
        raw_data_dir.mkdir(
            parents=True,
            exist_ok=True
        )
            
        output_path = raw_data_dir / "calendar_gdp.csv"
            
        grow_df_normalized.to_csv(
            output_path,
            index=False
        )
            
        print(f"GDP calendar saved to: {output_path}")
                    
        self.calendar = grow_df_normalized

        return grow_df_normalized
    
    #--- alternatives data collection ---
    


    


    def get_sf_fed_news_sentiment(
        self,
        save=True
    ):
        from io import BytesIO

        url = (
            "https://www.frbsf.org/"
            "wp-content/uploads/"
            "news_sentiment_data.xlsx"
        )

        print(
            "Downloading SF Fed Daily News Sentiment..."
        )

        response = self.session.get(
            url,
            timeout=self.timeout
        )

        response.raise_for_status()

        # --------------------------------------------------
        # Read the DATA sheet
        # --------------------------------------------------

        df = pd.read_excel(
            BytesIO(response.content),
            sheet_name="Data"
        )

        # --------------------------------------------------
        # Clean column names
        # --------------------------------------------------

        df = df.rename(
            columns={
                "date": "observation_date",
                "News Sentiment": "news_sentiment"
            }
        )

        df["observation_date"] = pd.to_datetime(
            df["observation_date"],
            errors="coerce"
        )

        df["news_sentiment"] = pd.to_numeric(
            df["news_sentiment"],
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
            .reset_index(drop=True)
        )

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        if save:

            project_root = (
                Path(__file__).resolve().parents[1]
            )

            raw_data_dir = (
                project_root
                / "data"
                / "raw"
                / "alternative"
            )

            raw_data_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            # Original source
            excel_path = (
                raw_data_dir
                / "sf_fed_news_sentiment.xlsx"
            )

            with open(excel_path, "wb") as f:

                f.write(
                    response.content
                )

            # Clean pandas dataframe
            pickle_path = (
                raw_data_dir
                / "sf_fed_news_sentiment.csv"
            )

            df.to_csv(
                pickle_path,
                index = False
            )

            print(
                f"News sentiment saved to: "
                f"{pickle_path}"
            )

        return df