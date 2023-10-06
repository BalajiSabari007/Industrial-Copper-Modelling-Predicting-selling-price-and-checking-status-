import pandas as pd 
import numpy as np
import streamlit as st 
import seaborn as sb 
import datetime
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#---------------------------------------------------------------------------------------------------------------------------------
st.title(":blue[Industrial Copper Modeling Price Prediction And Status Check]")

tab1, tab2 = st.tabs(["Predicting Selling Price", "checking Status"])

with tab1:

    status_list = ['Won','To be approved','Lost','Not lost for AM','Wonderful','Revised','Offered','Offerable']
    country_list = ['28', '32', '38', '78', '27', '30', '25', '77', '39', '40', '26', '84', '80', '79', '113', '89']
    item_list = ['W','S','Others','PL','WI','IPL']
    application_list = [10., 41., 28., 59., 15.,  4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79.,  3., 99.,  2.,  5., 39., 69., 70., 65., 58., 68.]
    product_ref = ['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', 
                     '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
                     '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
                     '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
                     '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']
     
    c1, c2, c3 = st.columns([2, 2, 2])

    with c1:
        Quantity_tons = st.text_input('Enter Quantity (Min:611728 & Max:1722207579) in tons')
        Thickness = st.text_input('Enter Thickness (Min:0.18 & Max:400)')
        Width = st.text_input('Enter Width (Min:1 & Max:2990)')

    with c2:
        Country = st.selectbox('Country', country_list)
        Status = st.selectbox('Status', status_list)
        item = st.selectbox('Item_type', item_list)

    with c3:
        application = st.selectbox('choose Application', application_list)
        product = st.selectbox('Product Reference', product_ref)
        order_date = st.date_input('Item_order_date', datetime.date(2023, 1, 1))
        delivery_date = st.date_input('Item_delivery_date', datetime.date(2023, 1, 1))

    with c1:
        st.write('')
        if st.button('PREDICT THE PRICE'):
            data = []
            with open('country.pkl', 'rb') as file:
                encode_country = pickle.load(file)
            with open('status.pkl', 'rb') as file:
                encode_status = pickle.load(file)
            with open('item.pkl', 'rb') as file:
                encode_item = pickle.load(file)
            with open('Stdscaling.pkl', 'rb') as file:
                scaled_data = pickle.load(file)
            with open('ExtraTreesRegressor.pkl', 'rb') as file:
                trained_model = pickle.load(file)

            transformed_country = encode_country.transform(country_list)
            c = None
            for i, j in zip(country_list, transformed_country):
                if Country == i:
                    c = j
                    break
                else:
                    st.error('Country not found')
                    exit()

            transformed_status = encode_status.transform(status_list)
            s = None
            for i, j in zip(status_list, transformed_status):
                if Status == i:
                    s = j
                    break
                else:
                    st.error('status not found')
                    exit()

            transformed_item = encode_item.transform(item_list)
            it = None
            for i, j in zip(item_list, transformed_item):
                if item == i:
                    it = j
                    break
                else:
                    st.error('item not found')
                    exit()

            order_date = datetime.strptime(str(order_date), format="%Y-%m-%d")
            delivery_date = datetime.strptime(str(delivery_date), format="%Y-%m-%d")
            day = delivery_date - order_date


            data.append(Quantity_tons)
            data.append(Thickness)
            data.append(Width)
            data.append(c)
            data.append(s)
            data.append(it)
            data.append(application)
            data.append(product)
            data.append(day.days)

            X = np.array(data).reshape(1, -1)
            predict_model = scaled_data.transform(X)
            price_predict = trained_model.predict(predict_model)
            print(f"Predicted Selling Price : green[Rs.]: green[{price_predict}]")

             
with tab2:
    country_clm = ['28', '32', '38', '78', '27', '30', '25', '77', '39', '40', '26', '84', '80', '79', '113', '89']
    item_clm = ['W','S','Others','PL','WI','IPL']
    application_clm = [10., 41., 28., 59., 15.,  4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79.,  3., 99.,  2.,  5., 39., 69., 70., 65., 58., 68.]
    product_clm = ['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', 
                     '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
                     '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
                     '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
                     '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

    c1, c2, c3 = st.columns([2, 2, 2])

    with c1:
        Quantity_cl = st.text_input('Enter Quantity (Min:611728 & Max:1722207579) in ton')
        Thickness_cl = st.text_input('Enter thickness (Min:0.18 & Max:400)')
        Width_cl = st.text_input('Enter width (Min:1 & Max:2990)')

    with c2:
        Country_cl = st.selectbox('Country_value', country_clm)
        item_cl = st.selectbox('Item_type_value', item_clm)
        selling_price = st.text_input('Enter Selling Price, (Min:1, Max:100001015)')

    with c3:
        application_cl = st.selectbox('choose-Application', application_clm)
        product_cl = st.selectbox('Product-Reference', product_clm)
        order_date_cl = st.date_input('order_date', datetime.date(2023, 1, 1))
        delivery_date_cl = st.date_input('delivery_date', datetime.date(2023, 1, 1))

    with c1:
        st.write('')
        if st.button('Ckeck Status'):
            data_cls = []
            with open('country.pkl', 'rb') as file:
                encoded_country = pickle.load(file)
            with open('status.pkl', 'rb') as file:
                encoded_status = pickle.load(file)
            with open('item.pkl', 'rb') as file:
                encoded_item = pickle.load(file)
            with open('Stdscaling.pkl', 'rb') as file:
                scaled_data_cl = pickle.load(file)
            with open('ExtraTreesRegressor.pkl', 'rb') as file:
                trained_model_cl = pickle.load(file)

            transformed_country_cl = encoded_country.transform(country_clm)
            en_c = None
            for i, j in zip(country_clm, transformed_country_cl):
                if Country_cl == i:
                    en_c = j
                    break
                else:
                    st.error('Country not found')
                    exit()
            
            
            transformed_item_cl = encoded_item.transform(item_clm)
            en_it = None
            for i, j in zip(item_clm, transformed_item_cl):
                if item_cl == i:
                    en_it = j
                    break
                else:
                    st.error("Item type not found.")
                    exit()   

            order_date_cl = datetime.strptime(str(order_date_cl), format="%Y-%m-%d")
            delivery_date_cl = datetime.strptime(str(delivery_date_cl), format="%Y-%m-%d")
            day_cl = delivery_date_cl - order_date_cl

            data_cls.append(Quantity_cl)
            data_cls.append(Thickness_cl)
            data_cls.append(Width_cl)
            data_cls.append(en_c)
            data_cls.append(en_it)
            data_cls.append(selling_price)
            data_cls.append(application_cl)
            data_cls.append(product_cl)
            data_cls.append(order_date_cl)
            data.clsappend(delivery_date_cl)


            X_cls = np.array(data_cls).reshape(1, -1)
            pred_model = scaled_data_cl.transform(X_cls)
            status_check = trained_model_cl.predict(pred_model)

            if status_check == 6:
                st.write(f'Predicted Status :  :green[Won]')
            else:
                st.write(f'Predicted Status :  :red[Lost]')


               
                




    
