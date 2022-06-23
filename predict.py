import datetime

import numpy as np

import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from plot_graphs import plot_chart, plot_best_fit_line


@st.experimental_memo
def get_prediction_input(df, d1, d2, country, prediction_for, predict_days):
    if prediction_for == 'Confirmed':
        prediction_for = 'NewConfirmed'
    if prediction_for == 'Deaths':
        prediction_for = 'NewDeaths'
    d1 = datetime.datetime(d1.year, d1.month, d1.day)
    d2 = datetime.datetime(d2.year, d2.month, d2.day)
    data = df[(df['Date'] >= d1) & (df['Date'] <= d2)]
    if country == 'Global':
        data = data.groupby(['Date'])[[prediction_for]].sum()
    else:
        data = data[data['Country'] == country].groupby(['Date', 'Country'])[
            [prediction_for]].sum()
    data = data[(~data[prediction_for] < 0)]
    data_count = len(data)
    x = np.arange(data_count)
    y = data.values
    days_to_predict = np.arange(data_count, data_count + predict_days)
    return x, y, days_to_predict


@st.cache(allow_output_mutation=True)
def linear_regression(x, y, days_to_predict, values):
    response = {}
    response['same_future_data'] = False
    x = x.reshape(-1, 1)
    Fx = days_to_predict.reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(x, y)
    Yp = reg.predict(x)
    Fp = reg.predict(Fx)

    response['fit_line_graph'] = plot_best_fit_line(x, y, Yp)
    response['predict_graph'] = plot_chart(Fx.flatten(), Fp.astype(int).flatten(),
                                           name="Prediction",
                                           color="Blue",
                                           title="Future Prediction for Confirmed Cases By days",
                                           yaxis_title="Predicted Confirm Cases"
                                           )
    response['confidence_score'] = str(round(reg.score(x, y) * 100, 2))
    if len(np.unique(Fp.astype(int).flatten()).tolist()) == 1:
        response['same_future_data'] = True
    return response


@st.cache(allow_output_mutation=True)
def polynomial_regression(x, y, days_to_predict, values):
    response = {}
    response['same_future_data'] = False
    x = x.reshape(-1, 1)
    Fx = days_to_predict.reshape(-1, 1)
    poly = PolynomialFeatures(degree=values['degree'])
    X = poly.fit_transform(x)
    FX = poly.fit_transform(Fx)
    reg_poly = LinearRegression()
    reg_poly.fit(X, y)
    Yp = reg_poly.predict(X)
    Fp = reg_poly.predict(FX)
    response['fit_line_graph'] = plot_best_fit_line(x, y, Yp)
    response['predict_graph'] = plot_chart(Fx.flatten(), Fp.astype(int).flatten(),
                                           name="Prediction",
                                           color="Blue",
                                           title="Future Prediction for Confirmed Cases By days",
                                           yaxis_title="Predicted Confirm Cases"
                                           )
    response['confidence_score'] = str(round(reg_poly.score(X, y) * 100, 2))
    if len(np.unique(Fp.astype(int).flatten()).tolist()) == 1:
        response['same_future_data'] = True
    return response


@st.cache(allow_output_mutation=True)
def support_vector_regression(x, y, days_to_predict, values):
    response = {}
    response['same_future_data'] = False
    x = x.reshape(-1, 1)
    Fx = days_to_predict.reshape(-1, 1)
    y = y.reshape(-1, 1)
    sc_x = StandardScaler()  # need to standardize the data to keep the data inside a particular range
    sc_y = StandardScaler()
    sc_fx = StandardScaler()
    Sx = sc_x.fit_transform(x)
    Sy = sc_y.fit_transform(y)
    Fxx = sc_fx.fit_transform(Fx)
    reg_svr = SVR(kernel='rbf')
    reg_svr.fit(Sx, Sy)  # ravel convert 2d array to 1d array
    Yp = reg_svr.predict(Sx)
    Fp = reg_svr.predict(Fxx)
    response['fit_line_graph'] = plot_best_fit_line(Sx, Sy, Yp.reshape(-1, 1))
    response['predict_graph'] = plot_chart(Fx.flatten(), Fp.astype(int).flatten(),
                                           name="Prediction",
                                           color="Blue",
                                           title="Future Prediction for Confirmed Cases By days",
                                           yaxis_title="Predicted Confirm Cases"
                                           )
    response['confidence_score'] = str(round(reg_svr.score(Sx, Sy) * 100, 2))
    if len(np.unique(Fp.astype(int).flatten()).tolist()) == 1:
        response['same_future_data'] = True
    return response


@st.cache(allow_output_mutation=True)
def decision_tree_regression(x, y, days_to_predict, values):
    response = {}
    response['same_future_data'] = False
    x = x.reshape(-1, 1)
    Fx = days_to_predict.reshape(-1, 1)
    reg_dec = DecisionTreeRegressor()
    reg_dec.fit(x, y)
    Yp = reg_dec.predict(x)
    Fp = reg_dec.predict(Fx)
    response['fit_line_graph'] = plot_best_fit_line(x, y, Yp.reshape(-1, 1))
    response['predict_graph'] = plot_chart(Fx.flatten(), Fp.astype(int).flatten(),
                                           name="Prediction",
                                           color="Blue",
                                           title="Future Prediction for Confirmed Cases By days",
                                           yaxis_title="Predicted Confirm Cases"
                                           )
    response['confidence_score'] = str(round(reg_dec.score(x, y) * 100, 2))
    if len(np.unique(Fp.astype(int).flatten()).tolist()) == 1:
        response['same_future_data'] = True

    return response


@st.cache(allow_output_mutation=True)
def random_forest_regression(x, y, days_to_predict, values):
    response = {}
    response['same_future_data'] = False
    x = x.reshape(-1, 1)
    Fx = days_to_predict.reshape(-1, 1)
    reg_rand = RandomForestRegressor(
        n_estimators=values['estimator'])  # n_estimator is the no.of tree, default is 100
    reg_rand.fit(x, y)
    Yp = reg_rand.predict(x)
    Fp = reg_rand.predict(Fx)
    response['fit_line_graph'] = plot_best_fit_line(x, y, Yp.reshape(-1, 1))
    response['predict_graph'] = plot_chart(Fx.flatten(), Fp.astype(int).flatten(),
                                           name="Prediction",
                                           color="Blue",
                                           title="Future Prediction for Confirmed Cases By days",
                                           yaxis_title="Predicted Confirm Cases"
                                           )
    response['confidence_score'] = str(round(reg_rand.score(x, y) * 100, 2))
    if len(np.unique(Fp.astype(int).flatten()).tolist()) == 1:
        response['same_future_data'] = True
    return response


@st.experimental_memo
def get_prediction_algo():
    predictions_algo = {
        'Linear Regression': linear_regression,
        'Polynomial Regression': polynomial_regression,
        'Support Vector Regression': support_vector_regression,
        'Decision Tree Regression': decision_tree_regression,
        'Random Forest Regression': random_forest_regression
    }
    return predictions_algo
