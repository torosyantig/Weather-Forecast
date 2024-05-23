import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Weather Forecast",
    page_icon="🌤",
    layout="centered",
    initial_sidebar_state="expanded",
)

model = joblib.load('temperature_predictor.pkl')
weather_data = joblib.load('weather_data.pkl')
cities_data = joblib.load('cities_data.pkl')
aggregated_predictions = joblib.load('aggregated_predictions.pkl')

def get_flag(country_code):
    offset = ord('🇦') - ord('A')
    return chr(ord(country_code[0]) + offset) + chr(ord(country_code[1]) + offset)

def celsius_to_fahrenheit(celsius):
    return celsius * 9/5 + 32

st.title("🌤 Weather Forecast 🌤")

# Create columns for layout
col1, col2 = st.columns([5, 1])

with col1:
    country = st.selectbox("🌍 Select Country", cities_data['country'].unique(), key="country")
    cities_in_country = cities_data[cities_data['country'] == country]['city_name']
    city = st.selectbox("🏙 Select City", cities_in_country, key="city")
    selected_date = st.date_input("📅 Select Date", key="date")

with col1:
    temp_unit = st.selectbox("Show weather in", ["Celsius", "Fahrenheit"], key="temp_unit")

country_code = cities_data[cities_data['country'] == country]['iso2'].values[0]
flag = get_flag(country_code)

st.subheader(f"{flag} Temperatutre for   {country} - {city} in {selected_date.strftime('%d.%m.%Y')}:")
#st.write(f"Temperature for {selected_date.strftime('%d.%m.%Y')}:")
#st.write(f"Selected Date: {selected_date.strftime('%d.%m.%Y')}")

month = selected_date.month
day = selected_date.day

if not weather_data[(weather_data['city_name'] == city) & (weather_data['date'] == pd.to_datetime(selected_date))].empty:
    avg_temp = weather_data[(weather_data['city_name'] == city) & (weather_data['date'] == pd.to_datetime(selected_date))]['avg_temp_c'].values[0]
    if temp_unit == "Fahrenheit":
        avg_temp = celsius_to_fahrenheit(avg_temp)
        st.write(f"🌡️ Avg Temperature = **{avg_temp:.2f} °F**")
    else:
        st.write(f"🌡️ Avg Temperature = **{avg_temp:.2f} °C**")
else:
    prediction = aggregated_predictions[(aggregated_predictions['city_name'] == city) & 
                                         (aggregated_predictions['month'] == month) & 
                                         (aggregated_predictions['day'] == day)]
    if not prediction.empty:
        avg_temp = prediction['avg_predicted_temp'].values[0]
        if temp_unit == "Fahrenheit":
            avg_temp = celsius_to_fahrenheit(avg_temp)
            st.write(f"🌡️ Avg Temperature = **{avg_temp:.2f} °F**")
        else:
            st.write(f"🌡️ Avg Temperature = **{avg_temp:.2f} °C**")
    else:
        st.write("❌ No data available for the selected date and city.")

st.subheader("Weather Forecast for the Next 7 Days:")

next_7_days = pd.date_range(selected_date, periods=7)
forecast = []

for date in next_7_days:
    month = date.month
    day = date.day
    prediction = aggregated_predictions[(aggregated_predictions['city_name'] == city) & 
                                         (aggregated_predictions['month'] == month) & 
                                         (aggregated_predictions['day'] == day)]
    if not prediction.empty:
        avg_temp = prediction['avg_predicted_temp'].values[0]
        if temp_unit == "Fahrenheit":
            avg_temp = celsius_to_fahrenheit(avg_temp)
        forecast.append((date.strftime("%d.%m.%Y"), date.strftime("%a"), f"{avg_temp:.2f} °{temp_unit[0]}"))
    else:
        forecast.append((date.strftime("%d.%m.%Y"), date.strftime("%a"), "No data"))

forecast_df = pd.DataFrame(forecast, columns=["Date", "Day", "Avg Temperature (°C)"])

header = pd.DataFrame([{
    "Date": f"{flag} {country}",
    "Day": city,
    "Avg Temperature (°C)": "Average Daily Temperature"
}])

forecast_df = pd.concat([header, forecast_df], ignore_index=True)

forecast_df = forecast_df.T

forecast_df.columns = forecast_df.iloc[0]
forecast_df = forecast_df[1:]

forecast_df.reset_index(drop=True, inplace=True)

st.write(forecast_df.style.set_table_styles(
    [{
        'selector': 'thead th',
        'props': [('background-color', '#4CAF50'), ('color', 'white'), ('font-size', '14px')]
    }, {
        'selector': 'tbody tr',
        'props': [('font-size', '12px')]
    }, {
        'selector': 'tbody tr:nth-child(odd)',
        'props': [('background-color', '#f2f2f2')]
    }]
))

if len(forecast_df.columns) > 1:
    dates = forecast_df.columns[1:]
    temps = forecast_df.iloc[1, 1:].str.replace(f' °{temp_unit[0]}', '').astype(float)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=temps,
        mode='lines+markers+text',
        text=[f'{temp:.2f}°{temp_unit[0]}' for temp in temps],
        textposition='top center',
        line=dict(color='royalblue', width=2),
        marker=dict(color='red', size=10)
    ))

    fig.update_layout(
        title={
            'text': 'Average Temperature Changes During the Week',
            'x': 0.5,
            'xanchor': 'center'
        },        xaxis_title='Date',
        yaxis_title=f'Avg Temperature (°{temp_unit[0]})',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    st.plotly_chart(fig)
else:
    st.write("❌ Insufficient data to plot the graph.")