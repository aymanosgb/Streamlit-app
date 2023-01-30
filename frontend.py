import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

st.set_page_config(layout="wide")
header = st.container()
dataset =  st.container()  
features = st.container()
model_training = st.container()

st.markdown(
    """""
    <style>
    .main {
        background-color: #F5F5F5;
        }
        </style>
        """,unsafe_allow_html=True
)

@st.cache
def get_data(filename):
    taxi_data = pd.read_csv(filename, dtype='unicode')
    return taxi_data
with header : 
    st.title("Welcome to my Data science project :white_heart:")
    st.subheader('by :blue[_Ayman Moumen_] ')
    st.text("In this project I look into the transactions of taxis on NYC")
    st.write("You can find the code [here](https://github.com/aymanosgb/streamlit-app) on my github. Feel free to check it out")

with dataset:
    st.header("NYC taxi dataset :")
    st.write("I found this dataset on this [Kaggle link](https://www.kaggle.com/datasets/anandaramg/taxi-trip-data-nyc) ")
    taxi_data = get_data("Data/uber.csv")
    st.write(taxi_data.head(10))
    
    st.subheader("Passenger count distribution in the taxi dataset")
    passenger_count_dist = pd.DataFrame(taxi_data["passenger_count"].value_counts())
    st.bar_chart(passenger_count_dist,use_container_width=True)

with features :
    st.header("The features i created")
    st.markdown("* **First feature** I created this feature combining longitude and latitude of the pickup location")
    st.markdown("* **Second feature** I created this feature because... I calculated it using ...")
    st.markdown(" **Objectives :** In this project we will use a random forest model to predict the fare of the taxi trips")
    st.markdown(" We will select the parameters of the model, train the mopdel then evaluate it with different metrics")


with model_training :
    st.header("Time to train the model")
    st.text("here you get to choose the hyperparameters of the model and see how the performance changes ")
    
    sel_col, disp_col = st.columns(2    )   
    
    max_depth = sel_col.slider("What should be the max_depth of the model?      ",min_value=10,max_value=100,value=10,step=10)
    
    n_estimators = sel_col.selectbox("how many trees should there be?",options=[100,200,100,"No limit"],index = 0)
    

    
    input_feature = sel_col.text_input("which feature should be used as the input feature?","PULocationID")
    
    sel_col.text("Here is a list of the feat features in our data:")
    sel_col.write(taxi_data.columns)
    
    if n_estimators == 'No limit':
        regr = RandomForestRegressor(max_depth=max_depth)
    else :
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
    
    x = taxi_data[[input_feature]]
    y = taxi_data["trip_distance"] 
    
    regr.fit(x,y)
    predictions = regr.predict(x)
    
    disp_col.subheader('Mean absolute error of the model is :')
    disp_col.write(round(mean_absolute_error(y,predictions)))
    
    disp_col.subheader('Mean squared error of the model is :')
    disp_col.write(round(mean_squared_error(y,predictions)))
    
    disp_col.subheader('R squared score of the model is :')
    disp_col.write(round(r2_score(y,predictions),4))
    
