import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt

df = pd.read_parquet('/Users/tigran/Downloads/archive/daily_weather.parquet')
cities_df = pd.read_csv('/Users/tigran/Downloads/archive/cities.csv')

df['date'] = pd.to_datetime(df['date'])

df['date'] = pd.to_datetime(df['date'])
df= df[(df['date'].dt.year >= 2013) & (df['date'].dt.year <= 2023)]

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

df['avg_temp_c'] = df['avg_temp_c'].fillna(df['avg_temp_c'].mean())
df['min_temp_c'] = df['min_temp_c'].fillna(df['min_temp_c'].mean())
df['max_temp_c'] = df['max_temp_c'].fillna(df['max_temp_c'].median())

X = df[['city_name', 'year', 'month', 'day', 'min_temp_c', 'max_temp_c']]
y = df['avg_temp_c']

categorical_features = ['city_name']
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

predictions_df = X_test.copy()
predictions_df['avg_predicted_temp'] = y_pred
predictions_df['avg_temp_actual'] = y_test.values

aggregated_predictions_df = predictions_df.groupby(['city_name', 'month', 'day']).agg({
    'avg_predicted_temp': 'mean',
    'avg_temp_actual': 'mean'
}).reset_index()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.title('Actual vs Predicted Temperature')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Diagonal line
plt.show()



joblib.dump(model, 'temperature_predictor.pkl')
joblib.dump(df, 'weather_data.pkl')
joblib.dump(cities_df, 'cities_data.pkl')

print("Model, data, and aggregated predictions saved as 'temperature_predictor.pkl', 'weather_data.pkl', 'cities_data.pkl', and 'aggregated_predictions.csv'")