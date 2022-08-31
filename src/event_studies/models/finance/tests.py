from event_studies.models.finance import MarketModel
import numpy as np
from typing import List

__all__ = ["calculate_aggregations", "j1", "j2", "rank_test",
           "generalized_sign_test"]


def calculate_aggregations(market_model: MarketModel,
                           event_sample: List[dict]) -> List[dict]:
    """TODO."""

    aggregations = []
    for event in event_sample:
        market_model.fit(firm_estimation_data=event["estimation_data"],
                         price_col=event["firm"],
                         date_col=event["date_col"])
        event_agg = market_model.abnormal_return_aggregations(event["start"],
                                                              event["end"],
                                                              firm_event_data=event["event_data"],
                                                              price_col=event["firm"],
                                                              date_col=event["date_col"])
        aggregations.append(event_agg)

    return aggregations


def j1(aggregations) -> float:
    """TODO."""

    cars = []
    var_cars = []
    for aggregation in aggregations:
        cars.append(aggregation["parametric"]["car"])
        var_cars.append(aggregation["parametric"]["var_car"])

    average_car = np.array(cars).mean()
    var_average_car = np.array(var_cars).mean() / len(aggregations)
    j1_value = average_car / np.sqrt(var_average_car)

    return j1_value


def j2(aggregations) -> float:
    """TODO."""

    scars = []
    n = len(aggregations)
    dfs = []
    for aggregation in aggregations:
        scars.append(aggregation["parametric"]["scar"])
        dfs.append(aggregation["parametric"]["scar_t_student_df"])

    msg = "different estimation window size for events not allowed!"
    assert all(x == dfs[0] for x in dfs), msg
    df = dfs[0]

    average_scar = np.array(scars).mean()
    normalization_constant = np.sqrt(n * (1 - 2/df))
    j2_value = normalization_constant * average_scar

    return j2_value


def rank_test(aggregations) -> float:
    """TODO."""

    event_ranks = []
    ws = []
    total_ranks = []
    for aggregation in aggregations:
        ws.append(aggregation["event_window_size"])
        total_ranks.append(len(aggregation["ar_total_midranks"]))
        event_ranks.append(aggregation["ar_total_midranks"][-ws[-1]:, 0])

    msg = "different event window size for events not allowed!"
    assert all(x == ws[0] for x in ws), msg

    msg = "different event+estimation window size for events not allowed!"
    assert all(x == total_ranks[0] for x in total_ranks), msg

    event_ranks = np.array(event_ranks)
    average_rank_on_event = event_ranks.mean()
    expected_rank = (total_ranks[0] + 1) / 2
    rank_variance = ((event_ranks.mean(axis=0) - expected_rank) ** 2).mean()
    rank_statistic = (np.sqrt(ws[0]) * (average_rank_on_event - expected_rank)
                      / np.sqrt(rank_variance))

    return rank_statistic


def generalized_sign_test(aggregations) -> float:
    """TODO."""

    proportions = []
    w = 0
    for aggregation in aggregations:
        proportions.append(aggregation["estimation_window_ar_sign"])
        if aggregation["parametric"]["car"] > 0.0:
            w += 1

    p = np.mean(proportions)
    n = len(proportions)
    generalized_sign_statistic = (w - n * p) / np.sqrt(n * p * (1-p))

    return generalized_sign_statistic
