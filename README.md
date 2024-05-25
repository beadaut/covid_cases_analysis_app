# Covid Cases Analysis App

This repository contains a Streamlit app for analyzing Covid cases in African countries. The app provides visualizations and insights based on the data.

## Installation

To run the app, you need to install the following packages:

- streamlit
- pandas
- plotly
- statsmodels
- geopandas

You can install these packages using pip:

```
pip install streamlit pandas plotly statsmodels geopandas
```

## Data

The app requires a data file, which should be placed in a folder called "data". You can download the data file from the following link:

- [Covid Data for African Countries](https://raw.githubusercontent.com/CodeForAfrica/covid19-in-africa/master/datasets/africa_historic_data.csv)

## Usage

To start the app, run the following command:

```
streamlit run app.py
```

Once the app is running, you can interact with it to explore the Covid cases data for African countries. The app provides the following features:

### History of Covid Cases by Countries

This section displays a scatter plot of Covid cases (Confirmed, Deaths, Recovered) over time for different African countries. You can select a specific case to interact with the plot.

### Map Plot of Covid Cases by Countries

This section shows a choropleth map plot of Covid cases for each African country. The color intensity represents the number of recovered cases.

### Daily Cases for Each Country

In this section, you can select a specific country and a case type (Confirmed, Deaths, Recovered) to view the daily cases plot. The plot shows the daily change in cases over time.

### Forecast for the Next 7 Days

The app also provides a forecast for the next 7 days of confirmed cases for the selected country. The forecast is based on the historical data using an AutoRegressive model.

## Conclusion

This app provides a comprehensive analysis of Covid cases in African countries. It offers various visualizations and insights to understand the trends and patterns in the data. Feel free to explore and interact with the app to gain valuable insights into the Covid situation in Africa.
