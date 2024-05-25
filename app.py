"""
- This is a starting script for a streamlit app.

- You need to install the following packages:
pip install streamlit pandas plotly statsmodels geopandas

- Create a folder called data and put the data file in it.

- Copy emoji from here: https://emojipedia.org/

"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.ar_model import AutoReg
from math import sqrt
from sklearn.metrics import mean_squared_error

st.title("🦠 Covid Data Analysis for African Countries")
st.write(
    "This is a app to show the analysis of covid cased in African Countries. The data is updated until March, 2023 and it shows information about Confirmed, Deaths and Recovered."
)


# Example: show sample data output
# Get dataset here
@st.cache_data  # this allows to save the data in a memory so that we don't have to load it again
def load_data():
    parse_dates = ["Date"]
    df = pd.read_csv(
        "https://raw.githubusercontent.com/CodeForAfrica/covid19-in-africa/master/datasets/africa_historic_data.csv",
        parse_dates=parse_dates,
    )
    return df


# show raw data
covid_history_africa = load_data()
# st.subheader("Raw data Display")
# st.write(covid_history_africa)

st.subheader("1.0 History of Covid Cases by Countries")
st.write(
    """This is the plot of R&D Spend vs Profit. It shows a positive correlation. 
    And this is a very long sentence that I am writing to see how it will look like in the app. 
    \n**Select a case to interact with the plot.**"""
)

# "Select a country", covid_history_africa["Country/Region"].unique()
selected_case = st.selectbox(
    "Select a case", ["Confirmed", "Deaths", "Recovered"], index=0
)
data = px.scatter(
    covid_history_africa,
    x="Date",
    y=selected_case,
    color="Country/Region",
    # title="Africa Covid History",
)
fig = go.Figure(data=data, layout={"type": "date"})
st.plotly_chart(fig)

st.write("---")

st.subheader("Map Plot of Covid Cases by Countries")
st.write(
    """
    This is a map plot (choropleth) of the covid cases showing each countries. 
    And this is a very long sentence that I am writing to see how it will look like in the app. 
    """
)

countries_grouped = (
    covid_history_africa.groupby("Country/Region").max().reset_index()
)
countries_df = px.data.gapminder().query("year==2007")
countries_and_iso = countries_df[["country", "iso_alpha"]]
countries_grouped = countries_grouped.merge(
    countries_and_iso, left_on="Country/Region", right_on="country"
)

data = px.choropleth(
    countries_grouped,
    locations="iso_alpha",
    color="Recovered",
    scope="africa",
)
fig = go.Figure(data=data)
st.plotly_chart(fig)

st.write("---")


def shift_list(l, n=1, fill=0):
    return ([fill] * n) + l[:-n]


# single out a country
st.subheader("Daily cases for each country")
st.write(
    """
    This is a map plot (choropleth) of the covid cases showing each countries. 
    And this is a very long sentence that I am writing to see how it will look like in the app. 
    """
)
col1, col2 = st.columns(2)
with col1:
    country = st.selectbox(
        "Select a country",
        covid_history_africa[
            "Country/Region"
        ].unique(),  # this is a list of all unique countries
        index=0,
        key="countries",
    )
with col2:
    selected_case = st.selectbox(
        "Select a case",
        ["Confirmed", "Deaths", "Recovered"],
        index=0,
        key="cases",
    )
selected_case_titles = {
    "Deaths": "Deaths Daily",
    "Confirmed": "Confirmed Daily",
    "Recovered": "Recovered Daily",
}

single_country_covid_history = covid_history_africa[
    covid_history_africa["Country/Region"] == country
]
single_country_covid_history = single_country_covid_history.sort_values("Date")
single_country_covid_history[selected_case_titles[selected_case]] = shift_list(
    single_country_covid_history[selected_case].tolist()
)
single_country_covid_history[selected_case_titles[selected_case]] = (
    single_country_covid_history[selected_case]
    - single_country_covid_history[selected_case_titles[selected_case]]
)

data = px.scatter(
    single_country_covid_history,
    x="Date",
    y=selected_case_titles[selected_case],
    # title="Nigeria Covid History",
)
fig = go.Figure(data=data, layout={"xaxis": {"type": "date"}})

st.plotly_chart(fig)


def predict(coef, history):
    yhat = coef[0]
    for i in range(1, len(coef)):
        yhat += coef[i] * history[-i]
    return yhat


# Let us predict some future values
data_size = len(single_country_covid_history)
train_fraction = int(0.8 * data_size)
train_data = single_country_covid_history[:train_fraction]
val_data = single_country_covid_history[train_fraction:]

window = 6
model = AutoReg(train_data["Confirmed"], lags=7)
model_fit = model.fit()

coef = model_fit.params
# walk forward over time steps in test
history = [train_data["Confirmed"].tolist()[i] for i in range(len(train_data))]
val_data_list = val_data["Confirmed"].tolist()
predictions = list()
for t in range(len(val_data)):
    yhat = predict(coef, history)
    obs = val_data_list[t]
    predictions.append(yhat)
    history.append(obs)
rmse = sqrt(mean_squared_error(val_data_list, predictions))


# forcast next 7 (or 14) days
history = [val_data["Confirmed"].tolist()[i] for i in range(len(val_data))]
predictions = list()
days = 7
for t in range(days):
    yhat = predict(coef, history)
    predictions.append(yhat)
    history.append(yhat)

val_data = val_data[:14]

data = {
    "Forecast": val_data["Confirmed"].tolist() + predictions,
    "t": list(range(len(val_data)))
    + list(range(len(val_data), len(val_data) + len(predictions))),
}
st.write(f"**Forecast [confirmed] for the next 7 days**: {country}")
fig = go.Figure()
data1 = px.scatter(
    data,
    y="Forecast",
    x="t",
    color=["History"] * len(val_data) + ["Forecast"] * len(predictions),
    title="Forecast [confirmed] for the next 7 days",
)
fig.add_traces(list(data1.select_traces()))
st.plotly_chart(fig)
