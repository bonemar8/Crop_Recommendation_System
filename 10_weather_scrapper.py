import requests
from bs4 import BeautifulSoup
import pandas as pd

#Fetching current tempeture, humidity from weather.com

url = "https://weather.com/tr-TR/weather/tenday/l/33d1e415eb66f3e1ab35c3add45fccf4512715d329edbd91c806a6957e123b49"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

day_containers = soup.find_all("div", attrs={"data-testid": "DetailsSummary"})

days = []
high_temps = []
low_temps = []
humidities = []

for container in day_containers[:10]:  
    day_name = container.find("h2", attrs={"data-testid": "daypartName"}).text.strip()
    high_temp = container.find("span", attrs={"class": "DetailsSummary--highTempValue--VHKaO"}).text.strip()
    low_temp = container.find("span", attrs={"class": "DetailsSummary--lowTempValue--ogrzb"}).text.strip()
    humidity = container.find("span", attrs={"data-testid": "PercentageValue"}).text.strip()
    
    days.append(day_name)
    high_temps.append(high_temp)
    low_temps.append(low_temp)
    humidities.append(humidity)
    

data = {
    "Day": days,
    "High Temperature": high_temps,
    "Low Temperature": low_temps,
    "Humidity": humidities
}
df = pd.DataFrame(data)

df.to_csv("10_day_weather_forecast.csv", index=False)

print("Weather data for the next 10 days:")
print(df)
