import requests
from bs4 import BeautifulSoup
import pandas as pd

#Fetching current tempeture, humidity from weather.com

url = "https://weather.com/tr-TR/weather/today/l/33d1e415eb66f3e1ab35c3add45fccf4512715d329edbd91c806a6957e123b49"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

temperature = soup.find("span", attrs={"data-testid": "TemperatureValue"}).text
humidity = soup.find("span", attrs={"data-testid": "PercentageValue"}).text

print(f"Humidity is: {humidity}")
print(f"Tempeture is: {temperature}")

weather = {
    "Temperature": [temperature],
    "Humidity": [humidity]
}

df = pd.DataFrame(weather)

df.to_csv("current_weather.csv", index=False)

print("Data has been saved  ")
