import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import PolynomialFeatures

from gaussian_features import GaussianFeatures


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


def predict_with_gaussian_features(time_frames: pd.Series, prices: pd.Series, num_predictions: int, n: int = 20):
    gauss_model = make_pipeline(GaussianFeatures(n), LinearRegression())

    gauss_model.fit(time_frames.values[:, np.newaxis], prices)

    last_time_frame: int = time_frames.values[-1]
    prediction_x = np.arange(last_time_frame + 1, last_time_frame + num_predictions)

    predictions = gauss_model.predict(prediction_x[:, np.newaxis])

    return np.array(list(zip(prediction_x, predictions))), gauss_model.predict, f'N: {n}'


def predict_with_polyfit(time_frames: pd.Series, prices: pd.Series, num_predictions: int,
                         degrees: int = 2, include_bias: bool = False):
    poly = PolynomialFeatures(degrees, include_bias=include_bias)
    linear = LinearRegression()
    steps = [('poly', poly), ('linear', linear)]
    poly_model = Pipeline(steps)
    poly_model.fit(time_frames[:, np.newaxis], prices)

    last_time_frame = time_frames.values[-1]
    prediction_x = np.arange(last_time_frame + 1, last_time_frame + num_predictions)
    prediction = poly_model.predict(np.reshape(prediction_x, (-1, 1)))

    return np.array(list(zip(prediction_x, prediction))), poly_model.predict, f'Degrees: {degrees}'


def predict_with_linear_model(time_frames: pd.Series, prices: pd.Series, num_predictions: int):
    shaped_time_frames: pd.Series = time_frames.values.reshape((-1, 1))
    shaped_prices: pd.Series = prices.values.reshape((-1, 1))
    model: LinearRegression = LinearRegression().fit(shaped_time_frames, shaped_prices)

    print(f'Model parameters: coefficients: {model.coef_}, intercepts: {model.intercept_}')

    last_time_frame = shaped_time_frames[-1]
    prediction_x = np.reshape(np.arange(last_time_frame + 1, last_time_frame + num_predictions), (-1, 1))

    prediction = model.predict(prediction_x)

    return np.array(list(zip(prediction_x,
                             prediction))), model.predict, f'Coefficient: {model.coef_[0][0]:.2f}, Intercept: {model.intercept_[0]:.2f}'


def plot(time_frames: pd.Series, prices: pd.Series, prediction, model_coefficients: np.ndarray, name: str,
         subtitle: str, annotate: bool = False):
    min_tf: float = time_frames.min()
    max_tf: float = time_frames.max()
    tf_stddev: float = time_frames.std()

    n: int = 100

    dummy_x: np.ndarray = np.linspace(min_tf - tf_stddev, max_tf + tf_stddev, n)

    plt.title(f'{name}\n({subtitle})')
    plt.plot(time_frames, prices, '.',
             dummy_x, model_coefficients(np.reshape(dummy_x, (-1, 1))), '-',
             prediction[:, 0], prediction[:, 1], '*')
    plt.xlabel('Time Frames')
    plt.ylabel('Bells')

    if annotate:
        for index in range(0, len(prediction)):
            time_frame = prediction[index][0]
            prediction_point = prediction[index][1]
            annotation_text: str = f'Next price ({time_frame}, {prediction_point:.1f})'
            plt.annotate(annotation_text, (time_frame, prediction_point))

    plt.show()


def get_option_print(option: str) -> str:
    return f'[{option[0]}]{option[1:]}'


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

    models = {'l': ('Linear Regression', predict_with_linear_model),
              'p': ('Polyfit', predict_with_polyfit),
              'g': ('Gaussian Features', predict_with_gaussian_features)}

    print('How would you like to predict the next turnip price?')

    for model_val in models.values():
        print(f'{get_option_print(model_val[0])}')

    model_input: str = input('> ')[0].lower()

    print('How many time frames (morning/evenings) would you like to see into the future?')
    num_predictions = int(input('> '))

    if model_input in models.keys():
        prediction, model, subtitle = models[model_input][1](time_frames, prices, num_predictions)
    else:
        print('Invalid model type.')
        return 0.0

    if model is not None:
        plot(time_frames, prices, prediction, model, models[model_input][0], subtitle)

    return prediction


if __name__ == '__main__':
    next_turnip_prices = predict_turnip_prices('data/prices_trystan.csv')
    print(f'Next Turnip Prices might be:\n{next_turnip_prices}')
