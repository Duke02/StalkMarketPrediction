import argparse
import datetime as dt
from sys import argv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
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


def parse_args():
    """
    Parses the arguments for this program.
    :return: The Namespace that contains the parsed arguments.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='Predict future turnip prices in Animal Crossing: New Horizons')
    parser.add_argument('-f', '--filepath', required=True, type=str,
                        help='The file that holds the previous turnip prices.')
    parser.add_argument('-M', '--model', required=False, type=str, choices=['l', 'p', 'g', 'r'], default=None,
                        help='The model to use. A prompt with minor explanations will be provided if this arg isn\'t '
                             'used.')
    parser.add_argument('-P', '--predictions', required=False, type=int, default=5,
                        help='The number of possible future turnip prices the model will output. Default is 5.')
    parser.add_argument('-N', '--gaussian_features', required=False, type=int, default=20,
                        help='The number of gaussian features to use with using the Gaussian-based models (r, '
                             'g). Default is 20.')
    parser.add_argument('-a', '--alpha', required=False, type=float, default=.1,
                        help='The alpha value to use for regularized models (r). Default value is .1.')
    parser.add_argument('-d', '--degrees', required=False, type=int, default=2,
                        help='The number of degrees to use for the polynomial based models (p). Default is 2.')
    parser.add_argument('-r', '--add_data', required=False, action='store_true',
                        help='Add random data to the model processing.')
    parser.add_argument('-I', '--iterations', required=False, type=int, default=0,
                        help='The number of iterations to add random data to the input.')

    return parser.parse_args(argv[1:])


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


def add_artificial_data(df: pd.DataFrame, num_iterations: int = 5) -> pd.DataFrame:
    values: np.ndarray = df.values
    output = values

    generator = np.random.default_rng()
    prices_std_dev: float = values[:, 1].std() / 6.
    time_frames_std_dev: float = values[:, 0].std() / 6.

    for i in range(num_iterations):
        multiplier: float = 1 if i % 2 == 0 else -1
        output = np.vstack((output, values + multiplier * generator.normal(scale=(time_frames_std_dev, prices_std_dev),
                                                                           size=values.shape)))

    return pd.DataFrame(data={'time_frame': output[:, 0], 'price': output[:, 1]})


def plot_learning_curve(model, time_frames: np.ndarray, prices: np.ndarray, cv: int = 5):
    """
    Plots the training/validation curve for the model with the given data.
    :param model: The model to be examined.
    :param time_frames: The independent variables of the data.
    :param prices: The dependent variables of the data.
    :param cv: The number of cross-validation folds to perform on the data.
    """
    # TODO: Input data is too small to perform cross-validation.
    n_samples_used, train_lc, val_lc = learning_curve(model, time_frames, prices, cv=cv)
    plt.plot(n_samples_used, np.mean(train_lc, 1), label='training score')
    plt.plot(n_samples_used, np.mean(val_lc, 1), label='validation score')
    plt.show()


def predict_with_regulated_gaussian(time_frames: pd.Series, prices: pd.Series, num_predictions: int,
                                    args: argparse.Namespace):
    """
    Performs a Regularized Gaussian Features regression analysis on the given data with the provided args.
    :param time_frames: The independent variables of the data. Shape = (N_Samples)
    :param prices: The dependent variables of the data. Shape = (N_Samples)
    :param num_predictions: The number of predictions to make for future time frames.
    :param args: The arguments for the program to alter hyper parameters.
    :return: The predictions of the data (Shape: (N_Samples, 2)), the model's prediction function, and the title for plotting.
    """
    # Since the data is in the range of 30-500, we can assume that the input data is dense.
    # So an L2 Norm using Ridge is probably best, rather than the L1 Norm, Lasso

    model = make_pipeline(GaussianFeatures(N=args.gaussian_features), Ridge(alpha=args.alpha))
    model.fit(time_frames.values[:, np.newaxis], prices)

    last_time_frame: int = time_frames.values[-1]
    prediction_x = np.arange(last_time_frame + 1, last_time_frame + num_predictions)
    predictions = model.predict(prediction_x[:, np.newaxis])

    return np.array(
        list(zip(prediction_x, predictions))), model.predict, f'N: {args.gaussian_features}, L2 a: {args.alpha}'


def predict_with_gaussian_features(time_frames: pd.Series, prices: pd.Series, num_predictions: int,
                                   args: argparse.Namespace):
    """
    Predicts the future turnip prices using a Gaussian Features model given the provided data and hyper parameters within args.
    :param time_frames: The time frames that have previously been recorded for the prices.
    :param prices: The prices for the turnip at those time frames.
    :param num_predictions: The number of predictions to make for future turnip prices.
    :param args: The args for the program that hold hyper parameters.
    :return: The predictions (Shape: (N_Predictions, 2)), the model's prediction function, and the title for plotting.
    """
    gauss_model = make_pipeline(GaussianFeatures(args.gaussian_features), LinearRegression())

    gauss_model.fit(time_frames.values[:, np.newaxis], prices)

    last_time_frame: int = time_frames.values[-1]
    prediction_x = np.arange(last_time_frame + 1, last_time_frame + num_predictions)

    predictions = gauss_model.predict(prediction_x[:, np.newaxis])

    return np.array(list(zip(prediction_x, predictions))), gauss_model.predict, f'N: {args.gaussian_features}'


def predict_with_polyfit(time_frames: pd.Series, prices: pd.Series, num_predictions: int, args: argparse.Namespace):
    poly_model = make_pipeline(PolynomialFeatures(degree=args.degrees, include_bias=False),
                               LinearRegression())
    poly_model.fit(time_frames[:, np.newaxis], prices)

    last_time_frame = time_frames.values[-1]
    prediction_x = np.arange(last_time_frame + 1, last_time_frame + num_predictions)
    prediction = poly_model.predict(prediction_x[:, np.newaxis])

    return np.array(list(zip(prediction_x, prediction))), poly_model.predict, f'Degrees: {args.degrees}'


def predict_with_linear_model(time_frames: pd.Series, prices: pd.Series, num_predictions: int,
                              args: argparse.Namespace):
    model: LinearRegression = LinearRegression().fit(time_frames.values[:, np.newaxis], prices[:, np.newaxis])

    print(f'Model parameters: coefficients: {model.coef_}, intercepts: {model.intercept_}')

    last_time_frame = time_frames.values[-1]
    prediction_x = np.reshape(np.arange(last_time_frame + 1, last_time_frame + num_predictions), (-1, 1))

    prediction = model.predict(prediction_x)

    return np.array(list(zip(prediction_x,
                             prediction))), model.predict, \
           f'Coefficient: {model.coef_[0][0]:.2f}, Intercept: {model.intercept_[0]:.2f} '


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
            annotation_text: str = f'({time_frame}, {prediction_point:.1f})'
            plt.annotate(annotation_text, (time_frame, prediction_point))

    plt.show()


def get_option_print(option: str) -> str:
    return f'[{option[0]}]{option[1:]}'


def predict_turnip_prices(args: argparse.Namespace) -> float:
    """
    Predicts the stalk market prices for the next time frame with the data at the provided file path.
    :param args: The arguments the user has put in.
    :return: The prediction for turnip prices in the next time frame.
    """
    df_raw: pd.DataFrame = get_data(args.filepath)
    df: pd.DataFrame = organize_data(df_raw)

    if args.add_data:
        df = add_artificial_data(df, args.iterations)

    time_frames = df['time_frame']
    prices = df['price']

    data_print = '\n'.join([f'{time_frames[i]:.5f}: {prices[i]:.5f}' for i in range(len(time_frames))])
    print(f'Input data: \n{data_print}')

    models = {'l': ('Linear Regression', predict_with_linear_model),
              'p': ('Polyfit', predict_with_polyfit),
              'g': ('Gaussian Features', predict_with_gaussian_features),
              'r': ('Regularized Gaussian', predict_with_regulated_gaussian)}

    if args.model is None:
        print('How would you like to predict the next turnip price?')

        for model_val in models.values():
            print(f'{get_option_print(model_val[0])}')

        model_input: str = input('> ')[0].lower()
    else:
        model_input: str = args.model

    num_predictions: int = args.predictions

    if model_input in models.keys():
        prediction, model, subtitle = models[model_input][1](time_frames, prices, num_predictions + 1, args)
    else:
        print('Invalid model type.')
        return 0.0

    if model is not None:
        plot(time_frames, prices, prediction, model, models[model_input][0], subtitle)

    return prediction


if __name__ == '__main__':
    arguments = parse_args()
    next_turnip_prices = predict_turnip_prices(arguments)
    print(f'Next Turnip Prices might be:\n{next_turnip_prices}')
