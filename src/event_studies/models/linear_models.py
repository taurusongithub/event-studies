"""TODO.

TODO.
"""

import numpy as np
from typing import NoReturn

__all__ = ["OLS"]


class OLS:
    """TODO."""

    def __init__(self) -> None:
        """TODO."""

        self.x_ = None
        self.y_ = None
        self.params_ = None
        self.residuals_ = None
        self.residuals_variance_ = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> NoReturn:
        """TODO."""

        self.x_ = x
        self.y_ = y

        n = len(x)
        msg = f"x, y shapes must be ({n}, 1)"
        assert (x.shape == (n, 1)) and (y.shape == (n, 1)), msg

        x_matrix = self._build_x_matrix(x)
        x_matrix_t = np.transpose(x_matrix)
        self.params_ = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_matrix_t,
                                                                   x_matrix)),
                                           x_matrix_t),
                                 y)
        self.residuals_ = self.forecast_error(x=x, y=y)
        residuals_variance_ = (np.matmul(np.transpose(self.residuals_),
                                         self.residuals_)
                               / (n - 2))
        self.residuals_variance_ = residuals_variance_[0, 0]

    @staticmethod
    def _build_x_matrix(x: np.ndarray) -> np.ndarray:
        """TODO."""

        n = len(x)
        msg = f"x shape must be ({n}, 1)"
        assert x.shape == (n, 1), msg
        x_ = np.concatenate((np.ones((n, 1)), x), axis=1)
        return x_

    def predict(self, x: np.ndarray) -> np.ndarray:
        """TODO."""

        y = np.matmul(self._build_x_matrix(x), self.params_)
        return y

    def forecast_error(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """TODO."""

        error = y - self.predict(x)
        return error

    def forecast_variance_matrix(self, x_new: np.ndarray) -> np.ndarray:
        """TODO."""

        n = len(x_new)
        x_matrix_new = self._build_x_matrix(x_new)
        x_matrix_new_t = np.transpose(x_matrix_new)
        x_matrix = self._build_x_matrix(self.x_)
        x_matrix_t = np.transpose(x_matrix)
        inv_x_t_x = np.linalg.inv(np.matmul(x_matrix_t, x_matrix))
        variance_matrix = (np.eye(n) * self.residuals_variance_
                           + (np.matmul(x_matrix_new,
                                        np.matmul(inv_x_t_x,
                                                  x_matrix_new_t))
                              * self.residuals_variance_))
        return variance_matrix
