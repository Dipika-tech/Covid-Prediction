import streamlit as st
import datetime
from predict import get_prediction_input, get_prediction_algo
import pandas as pd

st.set_page_config(page_title='Covid Prediction', page_icon=':memo:',
                   layout='wide', initial_sidebar_state='collapsed')

@st.experimental_memo
def get_day_wise_data():
    day_wise_df = pd.read_csv('data.csv')
    day_wise_df = day_wise_df[day_wise_df['Date'] != datetime.datetime.now().strftime("%Y-%m-%d")]
    day_wise_df = day_wise_df.sort_values(by=['Country'])
    day_wise_df['Date'] = day_wise_df['Date'].apply(
        lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))
    list_countries = tuple(['Global'] + day_wise_df['Country'].unique().tolist())
    return day_wise_df, list_countries


def prediction_page():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    predictions_algo = get_prediction_algo()

    st.title("Future Prediction Covid Data")
    placeholder = st.empty()
    start = datetime.datetime.now()

    # fetch data
    day_wise_df, list_countries = get_day_wise_data()
    # st.write()
    # frontend
    st.write("Input for prediction")
    l1, l2, l3, l4, l5, l6 = st.columns(6)
    d1 = l1.date_input(
        "Date From:",
        datetime.date(2022, 1, 1), min_value=datetime.date(2022, 1, 1),
        max_value=datetime.date.today(), key="d1")
    d2 = l2.date_input(
        "Date To:",
        datetime.date.today(), min_value=datetime.date(2022, 1, 1),
        max_value=datetime.date.today())
    prediction_for = l3.selectbox(
        'Prediction For', ('Confirmed', 'Deaths'))
    country = l4.selectbox(
        'Country',
        list_countries)
    predict_days = l5.number_input("Days to predict", value=10, min_value=1,
                                   help="Future prediction for how many days.")

    options = st.multiselect(
        'Select Prediction Algorithm',
        ['Linear Regression', 'Polynomial Regression', 'Support Vector Regression',
         'Decision Tree Regression', 'Random Forest Regression'],
        ['Linear Regression', 'Polynomial Regression'])

    x, y, days_to_predict = get_prediction_input(day_wise_df, d1, d2, country, prediction_for,
                                                 predict_days)

    # prediction algo frontend
    def get_fields(option):
        values = {}
        if option == 'Polynomial Regression':
            values['degree'] = st.number_input("Degree", value=5, min_value=1)
        if option == 'Random Forest Regression':
            values['estimator'] = st.number_input("estimator", value=100, min_value=100,
                                                  help="no of group tree")
        return values

    for i in range(len(options)):
        st.markdown("---")
        st.subheader(options[i])
        col1, col2, col3 = st.columns((2, 2, 1))
        with col1:
            fir_line = st.empty()
        with col2:
            predict_gr = st.empty()
        with col3:
            with st.expander("About"):
                st.write("""
                    The chart above shows some numbers I picked for you.
                    I rolled actual dice for these, so they're *guaranteed* to
                    be random.
                """)
            values = get_fields(options[i])
            st.markdown("##### **Days Trained:** " + str(x.shape[0]))
            confidence_score_line = st.empty()
            remark = st.empty()
        response = predictions_algo[options[i]](x, y,
                                                days_to_predict,
                                                values)
        fir_line.write(response['fit_line_graph'])
        predict_gr.write(response['predict_graph'])
        confidence_score_line.markdown(
            "##### **Confidence Score:** " + response['confidence_score'] + "%")
        if response['same_future_data']:
            remark.markdown(
                "##### **Remark:** Algorithm gets over valued,"
                " All the future prediction are same.Hence, this algo can't be consider.")

    elapsed_time = datetime.datetime.now() - start

    placeholder.markdown('<p class="update_date">Page loads in  ' + str(
        int(elapsed_time.total_seconds() * 1000)) + ' milisecond</p>', unsafe_allow_html=True)


prediction_page()
