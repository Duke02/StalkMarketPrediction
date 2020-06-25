import datetime as dt

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def is_morning(t: dt.time) -> bool:
    """
    Checks whether the given time is within the morning time frame for when Nook's Cranny is open.
    :param t: The time to check.
    :return: True if the given time is between 8am and noon. False otherwise.
    """
    opening_time = dt.time(hour=8, minute=0, second=0)
    noon = dt.time(hour=12, minute=0, second=0)
    return opening_time <= t <= noon


def is_evening(t: dt.time) -> bool:
    """
    Checks whether the given time is within the evening time frame when Nook's Cranny is open.
    :param t: The time to check.
    :return: True if the time is between noon and 10pm. False otherwise.
    """
    closing_time = dt.time(hour=22, minute=0, second=0)
    noon = dt.time(hour=12, minute=0, second=0)
    return noon <= t <= closing_time


def is_open(t: dt.time) -> bool:
    """
    Checks whether the given time is within the window of when Nook's Cranny is open or not.
    :param t: The time to check
    :return: True if the given time is between 8am and 10pm. False otherwise.
    """
    opening_time = dt.time(hour=8, minute=0, second=0)
    closing_time = dt.time(hour=22, minute=0, second=0)
    return opening_time <= t <= closing_time


def get_time_frame_dist(dt1: dt.datetime, dt2: dt.datetime) -> int:
    """
    Gets the number of time frames between the two dates and times.

    A time frame is either a morning or an evening within a single day. This is because
    turnip prices will change twice in a day, once when Nook's Cranny opens at 8am,
    and again at noon, when it is evening.

    :param d1: date that corresponds with t1
    :param t1: time that corresponds with d1
    :param d2: date that corresponds with t2
    :param t2: time that corresponds with d2
    :return: the number of time frames between the given dates and times.
    """
    print(f'dt1: {dt1}, type: {type(dt1)}')
    print(f'dt2: {dt2}, type: {type(dt2)}')

    diff_days = (dt2 - dt1).days
    print(f'Diff days: {diff_days}')
    output = diff_days * 2

    if is_evening(dt1.time()) != is_evening(dt2.time()):
        output += 1

    return output


def get_data(filename: str) -> pd.DataFrame:
    if filename.endswith('.csv'):
        return pd.read_csv(filename)
    raise ValueError('Unsupported data file. Can only support CSV\'s.')


def organize_data(df: pd.DataFrame) -> pd.DataFrame:
    prices = df['price']
    dates_and_times = pd.to_datetime(df['date_time'])

    print(dates_and_times)

    time_frames = []

    for index, dnt in enumerate(dates_and_times.values):
        print(f'{index}: {dnt}')
        if index == 0:
            time_frames += [0]
        else:
            dt1 = dates_and_times[index - 1].to_pydatetime()
            dt2 = pd.to_datetime(dnt).to_pydatetime()
            distance = get_time_frame_dist(dt1, dt2)
            time_frames += [time_frames[-1] + distance]

    return pd.DataFrame(data={'time_frame': time_frames, 'price': prices})


def predict_turnip_prices(filename: str) -> int:
    df_raw: pd.DataFrame = get_data(filename)
    df: pd.DataFrame = organize_data(df_raw)

    # This is the wrong shape.
    time_frames = np.array(df['time_frame']).reshape(-1, 1)
    prices = np.array(df['price']).reshape(-1, 1)

    print(f'Time Frames Size: {time_frames.size} | Shape: {time_frames.shape}')
    print(f'Prices Size: {prices.size} | Shape: {prices.shape}')

    model = LinearRegression().fit(df['time_frame'],
                                   df['price'])

    last_time_frame = df['time_frame'][-1]
    prediction = model.predict([last_time_frame])
    return prediction[0]


if __name__ == '__main__':
    next_turnip_price = predict_turnip_prices('data/prices_trystan.csv')
    print(f'Next Turnip Price might be {next_turnip_price}')
