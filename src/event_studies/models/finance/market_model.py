"""TODO.

TODO.
"""

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
from event_studies.models.linear_models import OLS
from typing import NoReturn, List, Optional
from scipy.stats import t as t_student

__all__ = ["MarketModel"]


class MarketModel:
    """TODO."""

    _date_col = "Date"

    def __init__(self, market_data: pd.DataFrame, market_index_col: str,
                 date_col: Optional[str] = None) -> None:
        """TODO."""

        if date_col is None:
            date_col = self._date_col

        self.market_index_col = market_index_col
        self._market_data = self._read_time_series(time_series=market_data,
                                                   data_col=market_index_col,
                                                   date_col=date_col)
        self.ols_ = None
        self.data_ = None
        self.firm_col_ = None

    @classmethod
    def _read_time_series(cls, time_series: pd.DataFrame, data_col: str,
                          date_col: str,
                          forbidden_names: List[str] = None) -> pd.DataFrame:
        """TODO."""

        if forbidden_names is None:
            forbidden_names = []

        forbidden_names.append(cls._date_col)
        msg = f"{data_col} is an invalid name for a data column"
        assert data_col not in forbidden_names, msg

        msg = f"{date_col} is not of datetime type"
        assert ptypes.is_datetime64_any_dtype(time_series[date_col]), msg
        msg = f"{data_col} is not numeric"
        assert ptypes.is_numeric_dtype(time_series[data_col]), msg

        time_series_ = time_series[[date_col, data_col]].copy()
        time_series_.rename(columns={date_col: cls._date_col}, inplace=True)

        return time_series_

    def _load_data(self, firm_estimation_data: pd.DataFrame, price_col: str,
                   date_col: str) -> NoReturn:
        """TODO."""

        self.data_ = self._data_plus_market(firm_data=firm_estimation_data,
                                            data_col=price_col,
                                            date_col=date_col)
        self.firm_col_ = price_col

    def fit(self, firm_estimation_data: pd.DataFrame, price_col: str,
            date_col: str = "Date") -> NoReturn:
        """TODO."""

        self._load_data(firm_estimation_data=firm_estimation_data,
                        price_col=price_col, date_col=date_col)
        self.ols_ = OLS()
        x = self.calculate_returns(prices=self.data_[self.market_index_col])
        y = self.calculate_returns(prices=self.data_[price_col])
        self.ols_.fit(x=x, y=y)

    def _data_plus_market(self, firm_data: pd.DataFrame, data_col: str,
                          date_col: str) -> pd.DataFrame:
        """TODO."""

        forbidden_names = [self.market_index_col]
        firm_data_ = self._read_time_series(time_series=firm_data,
                                            data_col=data_col,
                                            date_col=date_col,
                                            forbidden_names=forbidden_names)
        data_ = firm_data_.merge(self._market_data, on=self._date_col,
                                 how="left")\
                          .sort_values(by=self._date_col, ascending=True)\
                          .fillna(method="ffill", axis="columns")

        msg = "Data not available for the market portfolio on the given dates"
        assert data_[self.market_index_col].notna().all(), msg

        data_.set_index(self._date_col, inplace=True)
        return data_

    @staticmethod
    def calculate_returns(prices: pd.Series) -> np.ndarray:
        """TODO."""

        returns = np.log(((prices / prices.shift(1)).iloc[1:]).astype(float))
        n = len(returns)
        returns = returns.to_numpy().reshape(n, 1)

        return returns

    def _predict(self, event_data: pd.DataFrame) -> np.ndarray:
        """TODO."""

        market_index = event_data[self.market_index_col]
        event_market_returns = self.calculate_returns(prices=market_index)
        return self.ols_.predict(x=event_market_returns)

    def abnormal_returns(self, firm_event_data: pd.DataFrame, price_col: str,
                         date_col: str) -> np.ndarray:
        """TODO."""

        event_data = self._data_plus_market(firm_data=firm_event_data,
                                            data_col=price_col,
                                            date_col=date_col)
        market_index = event_data[self.market_index_col]
        market_returns = self.calculate_returns(market_index)
        firm_returns = self.calculate_returns(event_data[price_col])
        abnormal_returns = self.ols_.forecast_error(market_returns, firm_returns)
        return abnormal_returns

    def abnormal_returns_variance(self, firm_event_data: pd.DataFrame,
                                  price_col: str,
                                  date_col: str) -> np.ndarray:
        """TODO."""

        event_data = self._data_plus_market(firm_data=firm_event_data,
                                            data_col=price_col,
                                            date_col=date_col)
        market_index = event_data[self.market_index_col]
        market_returns = self.calculate_returns(market_index)
        variance_matrix = self.ols_.forecast_variance_matrix(market_returns)
        return variance_matrix

    def abnormal_return_aggregations(self, window_start: int, window_end: int,
                                     *args, **kwargs) -> dict:
        """TODO."""

        abnormal_returns = self.abnormal_returns(*args, **kwargs)
        abnormal_returns_variance = self.abnormal_returns_variance(*args,
                                                                   **kwargs)
        l2 = len(abnormal_returns)
        gamma = np.concatenate((np.zeros((window_start, 1)),
                                np.ones((window_end - window_start + 1, 1)),
                                np.zeros((l2 - window_end - 1, 1))), axis=0)
        car = np.matmul(np.transpose(gamma), abnormal_returns)[0, 0]
        var_car = np.matmul(np.transpose(gamma),
                            np.matmul(abnormal_returns_variance, gamma))[0, 0]
        scar = car / np.sqrt(var_car)
        aggregations = {"car": car, "scar": scar, "var_car": var_car,
                        "scar_t_student_df": (len(self.ols_.x_) - 2)}
        scar_distribution = t_student(aggregations["scar_t_student_df"])
        aggregations["p_value_two_sided"] = scar_distribution.sf(abs(scar)) * 2

        return aggregations
