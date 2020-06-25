import datetime as dt

import matplotlib.pyplot as plt
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

    :param dt1: The datetime that's before dt2
    :param dt2: The datetime that's after dt1
    :return: the number of time frames between the given dates and times.
    """
    diff_days = (dt2 - dt1).days
    output = diff_days * 2

    if is_evening(dt1.time()) != is_evening(dt2.time()):
        output += 1

    return output


def get_data(filepath: str) -> pd.DataFrame:
    """
    Gets the data at the provided file path.
    :param filepath: The file path that holds the data to be loaded into the program.
    :return: The data frame holding the turnip prices and datetimes for those prices.
    """
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    raise ValueError('Unsupported data file. Can only support CSV\'s.')


def organize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Organizes and cleans the provided data to use time frames, rather than datetimes as the independent variable.
    :param df: The raw data frame.
    :return: The cleaned up data frame.
    """
    prices = df['price']
    dates_and_times = pd.to_datetime(df['date_time'])

    time_frames = []

    for index, dnt in enumerate(dates_and_times.values):
        if index == 0:
            time_frames += [0]
        else:
            dt1 = dates_and_times[index - 1].to_pydatetime()
            dt2 = pd.to_datetime(dnt).to_pydatetime()
            distance = get_time_frame_dist(dt1, dt2)
            time_frames += [time_frames[-1] + distance]

    return pd.DataFrame(data={'time_frame': time_frames, 'price': prices})


def predict_with_linear_model(time_frames: pd.Series, prices: pd.Series) -> float:
    shaped_time_frames: pd.Series = time_frames.values.reshape((-1, 1))
    shaped_prices: pd.Series = prices.values.reshape((-1, 1))
    model: LinearRegression = LinearRegression().fit(shaped_time_frames, shaped_prices)

    print(f'Model parameters: coefficients: {model.coef_}, intercepts: {model.intercept_}')

    coefficients = np.poly1d([model.coef_[0][0], model.intercept_[0]])

    next_time_frame = shaped_time_frames[-1] + 1
    prediction = model.predict([next_time_frame])
    return [next_time_frame, prediction], coefficients


def predict_with_polyfit(time_frames: pd.Series, prices: pd.Series):
    shaped_time_frames: pd.Series = time_frames.values.reshape(-1)
    shaped_prices: pd.Series = prices.values.reshape(-1)

    degrees: int = 2
    poly_coefficients = np.polyfit(shaped_time_frames, shaped_prices, degrees)

    print(f'Polynomial coefficients: {poly_coefficients}')

    poly = np.poly1d(poly_coefficients)

    next_time_frame = shaped_time_frames[-1] + 1
    prediction = poly(next_time_frame)
    return [next_time_frame, prediction], poly


def plot(time_frames: pd.Series, prices: pd.Series, prediction, model_coefficients: np.ndarray):
    min_tf: float = time_frames.min()
    max_tf: float = time_frames.max()
    tf_stddev: float = time_frames.std()

    n: int = 100

    dummy_x: np.ndarray = np.linspace(min_tf - tf_stddev, max_tf + tf_stddev, n)

    plt.plot(time_frames, prices, '.',
             dummy_x, model_coefficients(dummy_x), '-',
             prediction[0], prediction[1], '*')
    plt.show()


def predict_turnip_prices(filepath: str) -> float:
    """
    Predicts the stalk market prices for the next time frame with the data at the provided file path.
    :param filepath: The path to the data file.
    :return: The prediction for turnip prices in the next time frame.
    """
    df_raw: pd.DataFrame = get_data(filepath)
    df: pd.DataFrame = organize_data(df_raw)

    time_frames = df['time_frame']
    prices = df['price']

    data_print = '\n'.join([f'{time_frames[i]}: {prices[i]}' for i in range(len(time_frames))])
    print(f'Input data: \n{data_print}')

    print('How would you like to predict the next turnip price?')
    model_input: str = input('[L]inear Regression, [P]olyfit\n')[0].lower()

    if model_input == 'l':
        prediction, model = predict_with_linear_model(time_frames, prices)
    elif model_input == 'p':
        prediction, model = predict_with_polyfit(time_frames, prices)
    else:
        print('Invalid model type.')
        return 0.0

    plot(time_frames, prices, prediction, model)

    return prediction


if __name__ == '__main__':
    next_turnip_price = predict_turnip_prices('data/prices_trystan.csv')
    print(f'Next Turnip Price might be {next_turnip_price}')
