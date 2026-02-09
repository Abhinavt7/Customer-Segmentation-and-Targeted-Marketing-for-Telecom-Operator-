import os
import sys
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(layout="wide")

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.preprocessing import encode_categorical

@st.cache_resource
def load_models():
    model_path = os.path.join(ROOT, 'models', 'kmeans_model.pkl')
    scaler_path = os.path.join(ROOT, 'models', 'scaler.pkl')
    kmeans = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return kmeans, scaler

kmeans_model, scaler = load_models()

st.sidebar.subheader('Customer Segmentation and Targeted Marketing for Telecom Operator')
st.sidebar.markdown('---')

page = st.sidebar.radio('Go To:',['Enter Customer Info','Cluster Information'],index=0)

if page == 'Enter Customer Info':
    st.header('Customer Behavior Clustering')
    option = st.selectbox('Select input method:', ['Manual Individual Entry', 'Upload CSV file'])

    if option == 'Manual Individual Entry':
        st.subheader('Manual Individual Entry of customer information')
        st.write('Enter values for all required fields (21 columns).')
        with st.form("manual_form"):
            DEVICETYPE = st.selectbox("Type of Device", ["Smartphone", "Voice Centric", "Feature Phone"], index=0)
            ISDEVICE3GENABLED = st.selectbox("Device 3G Enabled", ["Yes", "No"], index=0)
            VASSUBSCRIBERFLAG = st.selectbox("Subscribed to Value Added Services", ["Yes", "No"], index=1)
            ISDEVICEDATAENABLED = st.selectbox("Device Data Enabled", ["Yes", "No"], index=0)
            AGEONNETWORK = st.number_input("Age on the Network (in days)", value=600.0, format="%.2f")
            SUBSRIBERLASTBALANCE = st.number_input("Subscriber Last Balance", value=0.5, format="%.5f")
            last_app_used = st.text_input("last used (date or label)", value="2023-04-15")
            CNTCHURND1_30 = st.number_input("Churn Related events (last 30 days)", value=2.0, format="%.1f")
            HANDSETCHANGESFLAGD1_30 = st.number_input("Changed Device in last 30 days", value=1.0, format="%.1f")
            DEVICENETWORK = st.text_input("Device Network", value="4G")
            DEVICEMODELC = st.text_input("Device Model", value="Tecno Telecom")
            DEVICEDUALSIMFLAG = st.number_input("Dual SIM Flag", value=2.0, format="%.1f")
            ARPUD1_30 = st.number_input("Average Revenue Per User (last 30 days)", value=500.0, format="%.2f")
            SUMVCEREVCD1_30 = st.number_input("Voice Revenue (last 30 days)", value=25.0, format="%.2f")
            SUMDATAREVCD1_30 = st.number_input("Data Revenue (last 30 days)", value=1.0, format="%.2f")
            SUMDATAUSGCD1_30 = st.number_input("Data Usage (last 30 days)", value=5000000.0, format="%.1f")
            SUMVOICEUSGCD1_30 = st.number_input("Voice Usage (last 30 days)", value=1500.0, format="%.1f")
            SUMDATAUSG4GCD1_15 = st.number_input("Data Usage on 4G (last 15 days)", value=2500000.0, format="%.1f")
            SUMDATAUSG4GCD15_30 = st.number_input("Data Usage on 4G (last 15-30 days)", value=2500000.0, format="%.1f")
            SMARTPHONEFLAG = st.selectbox("Is Smartphone", ["Yes", "No"], index=0)
            MAINACTBAL1 = st.number_input("Main Account Balance", value=0.5, format="%.5f")

            submit = st.form_submit_button("Predict")

        if submit:
            record = pd.DataFrame({
                'DEVICETYPE': [DEVICETYPE],
                'ISDEVICE3GENABLED': [ISDEVICE3GENABLED],
                'VASSUBSCRIBERFLAG': [VASSUBSCRIBERFLAG],
                'ISDEVICEDATAENABLED': [ISDEVICEDATAENABLED],
                'AGEONNETWORK': [AGEONNETWORK],
                'SUBSRIBERLASTBALANCE': [SUBSRIBERLASTBALANCE],
                'last_app_used': [last_app_used],
                'CNTCHURND1_30': [CNTCHURND1_30],
                'HANDSETCHANGESFLAGD1_30': [HANDSETCHANGESFLAGD1_30],
                'DEVICENETWORK': [DEVICENETWORK],
                'DEVICEMODELC': [DEVICEMODELC],
                'DEVICEDUALSIMFLAG': [DEVICEDUALSIMFLAG],
                'ARPUD1_30': [ARPUD1_30],
                'SUMVCEREVCD1_30': [SUMVCEREVCD1_30],
                'SUMDATAREVCD1_30': [SUMDATAREVCD1_30],
                'SUMDATAUSGCD1_30': [SUMDATAUSGCD1_30],
                'SUMVOICEUSGCD1_30': [SUMVOICEUSGCD1_30],
                'SUMDATAUSG4GCD1_15': [SUMDATAUSG4GCD1_15],
                'SUMDATAUSG4GCD15_30': [SUMDATAUSG4GCD15_30],
                'SMARTPHONEFLAG': [SMARTPHONEFLAG],
                'MAINACTBAL1': [MAINACTBAL1]
            })

            st.write(record)
            enc = encode_categorical(record)
            scaled = scaler.transform(enc)
            pred = kmeans_model.predict(scaled)
            st.success(f"Customer Behaviour of Cluster: {pred[0]}")
            st.text('Check the "Cluster Information" page for more details on the predicted cluster.')

    if option == 'Upload CSV file':
        st.subheader('Uploading CSV file will be updated soon')

elif page == 'Cluster Information':

    left,right = st.columns([1.2,1])
    with left:
        st.subheader('Cluster Information')
        st.markdown("---")
        st.markdown('__ABBREVIATIONS__')
        st.text('SUMDATAREVCD1_30    -> Total data revenue in last 30 days'
                '\nSUMDATAUSGCD1_30  -> Total data usage in last 30 days'
                '\nSUMVOICEUSGCD1_30 -> Total voice usage in last 30 days'
                '\nARPUD1_30         -> Average revenue per user in last 30 days'
                '\nSUMVCEREVCD1_30   -> Total voice revenue in last 30 days'
                '\nAGEONNETWORK       -> Age of the customer on the network (in days)')
        st.markdown("---")
        st.markdown('__CLUSTER PROFILES__')
        st.text('Cluster 0 – “High data usage high data revenue"\n'
                 '- Above average ARPU, low voice usage n revenue\n'
                 '- Highest ARPU, high main account balance, long tenure, mostly smartphone users.\n'
                '\nCluster 1 – “Strong voice usage and voice revenue”\n'
                ' - Moderate ARPU, low data usage\n'
                ' - Low tenure, high 4G/data usage, low voice usage, mostly new to the network.\n'
                '\nCluster 2 – “Balanced across all features ”\n'
                ' - Moderate ARPU, Voice and Data usage\n'
                ' - Frequent zero-balance days, low revenue, short tenure, many feature phone users.\n'
                '\nCluster 3 – “Low data usage and Revenue”\n'
                ' - Low ARPU, Higher voice usage\n'
                ' - Above average in voice calls, moderate balance, low data usage, mid-high tenure.\n'
                '\nCluster 4 – “High voice usage and Similar Revenue to Cluser 0”\n'
                ' - Moderate Voice usage , better ARPU than average, long tenure, mostly smartphone users.\n'
                ' - Premium data users—engaged, high‑spending customers who are valuable for upselling.\n'
                )
    with right:
        st.image(os.path.join(os.getcwd(),"..","visualizations","cluster_radar.png"),caption="Customer Behavior Clusters ",width=630)#,height=400)
