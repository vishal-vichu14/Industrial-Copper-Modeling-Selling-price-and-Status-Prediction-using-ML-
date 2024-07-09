import streamlit as st
import numpy as np
import joblib
import pandas as pd
from datetime import date
from streamlit_option_menu import option_menu  # Assuming this is a custom module

# Constants and dictionaries
COUNTRY_VALUES = [25.0, 26.0, 27.0, 28.0, 30.0, 32.0, 38.0, 39.0, 40.0, 77.0,
                  78.0, 79.0, 80.0, 84.0, 89.0, 107.0, 113.0]

STATUS_VALUES = ['Won', 'Lost', 'Draft', 'To be approved', 'Not lost for AM',
                 'Wonderful', 'Revised', 'Offered', 'Offerable']
STATUS_DICT = {'Lost': 0, 'Won': 1, 'Draft': 2, 'To be approved': 3, 'Not lost for AM': 4,
               'Wonderful': 5, 'Revised': 6, 'Offered': 7, 'Offerable': 8}

ITEM_TYPE_VALUES = ['W', 'WI', 'S', 'PL', 'IPL', 'SLAWR', 'Others']
ITEM_TYPE_DICT = {'W': 5.0, 'WI': 6.0, 'S': 3.0, 'Others': 1.0, 'PL': 2.0, 'IPL': 0.0, 'SLAWR': 4.0}

APPLICATION_VALUES = [2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0,
                      27.0, 28.0, 29.0, 38.0, 39.0, 40.0, 41.0, 42.0, 56.0, 58.0,
                      59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0]

PRODUCT_REF_VALUES = [611728, 611733, 611993, 628112, 628117, 628377, 640400,
                      640405, 640665, 164141591, 164336407, 164337175, 929423819,
                      1282007633, 1332077137, 1665572032, 1665572374, 1665584320,
                      1665584642, 1665584662, 1668701376, 1668701698, 1668701718,
                      1668701725, 1670798778, 1671863738, 1671876026, 1690738206,
                      1690738219, 1693867550, 1693867563, 1721130331, 1722207579]

# Streamlit UI setup
st.markdown('<h1 style="color: red;">Airbnb Analysis</h1>', unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu("Prediction", ["Selling Price", 'Status'],
                           icons=['currency-dollar', 'clipboard-data-fill'], menu_icon="robot", default_index=0)

if selected == "Selling Price":
    st.subheader("Selling Price Prediction")

    # Input fields
    item_date = st.date_input('Item Date', date(2020, 7, 1), min_value=date(2020, 7, 1), max_value=date(2021, 5, 31))
    quantity_log = st.text_input('Quantity Tons (Min: 0.001 & Max: 40)')
    country = st.selectbox('Country', COUNTRY_VALUES)
    item_type = st.selectbox('Item Type', ITEM_TYPE_VALUES)
    thickness_log = st.text_input('Thickness (Min: 0.1 & Max: 10)')
    product_ref = st.selectbox('Product Ref', PRODUCT_REF_VALUES)
    delivery_date = st.date_input('Delivery Date', date(2020, 8, 1), min_value=date(2020, 8, 1),
                                  max_value=date(2022, 5, 28))
    customer = st.text_input('Customer ID (Min: 12458000 & Max: 2147484000)')
    status = st.selectbox('Status', STATUS_VALUES)
    application = st.selectbox('Application', APPLICATION_VALUES)
    width = st.text_input('Width (Min: 0.1 & Max: 3000)')

    # Calculate derived feature
    diff_date = (delivery_date - item_date).days

    # Button to submit
    if st.button('Submit'):
        random_forest_model = joblib.load('random_forest_model.pkl')
        user_data = np.array([[int(customer), float(country), STATUS_DICT[status], ITEM_TYPE_DICT[item_type],
                               float(application), float(width), float(product_ref),
                               np.log(float(quantity_log)), np.log(float(thickness_log)),
                               diff_date, item_date.day, item_date.month, item_date.year]])

        df = pd.DataFrame(data=user_data, columns=['customer', 'country', 'status', 'item_type', 'application',
                                                   'width', 'product_ref', 'quantity_log', 'thickness_log',
                                                   'diff_date', 'item_day', 'item_month', 'item_year'])

        y_pred = random_forest_model.predict(user_data)
        selling_price = np.exp(y_pred[0])
        selling_price = round(selling_price, 2)

        st.subheader("Data Entered")
        st.write(df)
        st.subheader(f"Predicted Selling Price: {selling_price}")

elif selected == "Status":
    st.subheader("Status Prediction")

    # Input fields
    item_datec = st.date_input('Item Date', date(2020, 7, 1), min_value=date(2020, 7, 1), max_value=date(2021, 5, 31))
    quantity_logc = st.text_input('Quantity Tons (Min: 0.001 & Max: 40)')
    sellingc = st.text_input("Selling price (Min: 0 & Max: 10000000)")
    countryc = st.selectbox('Country', COUNTRY_VALUES)
    item_typec = st.selectbox('Item Type', ITEM_TYPE_VALUES)
    thickness_logc = st.text_input('Thickness (Min: 0.1 & Max: 10)')
    product_refc = st.selectbox('Product Ref', PRODUCT_REF_VALUES)
    delivery_datec = st.date_input('Delivery Date', date(2019, 8, 1), min_value=date(2019, 8, 1),
                                   max_value=date(2022, 5, 28))
    customerc = st.text_input('Customer ID (Min: 12458000 & Max: 2147484000)')
    applicationc = st.selectbox('Application', APPLICATION_VALUES)
    widthc = st.text_input('Width (Min: 0.1 & Max: 3000)')

    # Calculate derived feature
    diff_datec = (delivery_datec - item_datec).days

    # Button to submit
    if st.button('Submit'):
        Extra_tree = joblib.load('Extra_trees.pkl')
        user_datac = np.array([[int(customerc), float(countryc), sellingc, ITEM_TYPE_DICT[item_typec],
                                float(applicationc), float(widthc), float(product_refc),
                                np.log(float(quantity_logc)), np.log(float(thickness_logc)),
                                diff_datec, item_datec.day, item_datec.month, item_datec.year]])

        df = pd.DataFrame(data=user_datac, columns=['customer', 'country', 'selling_price', 'item_type', 'application',
                                                    'width', 'product_ref', 'quantity_log', 'thickness_log',
                                                    'diff_date', 'item_day', 'item_month', 'item_year'])

        y_pred = Extra_tree.predict(user_datac)

        st.subheader("Data Entered")
        st.write(df)

        if y_pred == 1:
            st.subheader("Predicted Status:")
            st.markdown("<span style='color:green'>Won</span>", unsafe_allow_html=True)
        elif y_pred == 0:
            st.subheader("Predicted Status:")
            st.markdown("<span style='color:red'>Lost</span>", unsafe_allow_html=True)
