import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import pyowm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import EllipticEnvelope

st.set_page_config(layout="wide")

# Ввод API ключа
api_key = st.text_input("Введите ваш API ключ OpenWeatherMap:")

# Загрузка исторических данных
uploaded_file = st.file_uploader("Загрузите файл с историческими данными:", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    print(df.columns)
else:
    st.write("Пожалуйста, загрузите файл.")

# Выбор города
cities = ["Москва", "Берлин", "Каир", "Пекин"]  # Можно расширить список городов
selected_city = st.selectbox("Выберите город:", cities)


# Получение текущих погодных данных
"""
    23/10/2022 в 14:40 запросы на api.openweathermap.org из рф по https начали дропаться с таймаутом. порт 80 работает, но авторизация происходит по API ключу который отсылается как часть запроса, то-есть http это не вариант.
"""
def get_current_weather(api_key):
    if api_key:
        try:
            owm = pyowm.OWM(api_key)
            mgr = owm.weather_manager()
            observation = mgr.weather_at_place(selected_city)
            weather = observation.weather
            return weather.temperature('celsius')['temp']
        except Exception as e:
            st.error(f"Произошла ошибка при получении данных: {e}")
            return None
    else:
        st.warning("API ключ не указан. Текущие данные о погоде недоступны.")
        return None

current_temp = get_current_weather(api_key)

# Описательная статистика
if current_temp is not None:
    st.write(f"Текущая температура в {selected_city}: {current_temp:.1f}°C")


def detect_anomalies(data):
    clf = EllipticEnvelope(contamination=0.01)  # contamination — доля предполагаемых аномалий
    predictions = clf.fit_predict(data)
    anomalies = np.where(predictions == -1)[0]  # Аномалии обозначаются как '-1'
    return anomalies


# Временной ряд температур с аномалиями
print("IN", df.columns)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d")
anomalies = detect_anomalies(df["temperature"].values.reshape(-1, 1))
plt.figure(figsize=(10,6))
plt.plot(df["timestamp"], df["temperature"], label='Температура')
plt.scatter(df["timestamp"][anomalies], df["temperature"][anomalies], color='red', label='Аномалии')
plt.title(f"Временной ряд температур для {selected_city}")
plt.xlabel("Дата")
plt.ylabel("Температура (°C)")
plt.legend()
st.pyplot(plt.gcf())

# Сезонные профили
seasons = ['Зима', 'Весна', 'Лето', 'Осень']
month_to_season = {
    12: 'Зима',
    1: 'Зима',
    2: 'Зима',
    3: 'Весна',
    4: 'Весна',
    5: 'Весна',
    6: 'Лето',
    7: 'Лето',
    8: 'Лето',
    9: 'Осень',
    10: 'Осень',
    11: 'Осень'
}
df['Season'] = df['timestamp'].dt.month.map(month_to_season)
grouped_by_season = df.groupby('Season')
for season in seasons:
    group = grouped_by_season.get_group(season)
    mean_temp = group['temperature'].mean()
    std_dev = group['temperature'].std()
    st.write(f"{season}: Средняя температура {mean_temp:.1f}°C, Стандартное отклонение {std_dev:.1f}°C")

# Проверка нормальной температуры для текущего сезона
if current_temp is not None:
    current_month = datetime.now().month
    current_season = month_to_season[current_month]
    normal_range = grouped_by_season.get_group(current_season)['temperature'].quantile([0.25, 0.75])
    if normal_range[0.25] <= current_temp <= normal_range[0.75]:
        st.success(f"Температура {current_temp:.1f}°C является нормальной для {current_season}.")
    else:
        st.warning(f"Температура {current_temp:.1f}°C выходит за рамки нормального диапазона для {current_season}.")

# Функция обнаружения аномалий
def detect_anomalies(data):
    clf = EllipticEnvelope(contamination=0.01)
    predictions = clf.fit_predict(data)
    anomalies = np.where(predictions == -1)[0]
    return anomalies