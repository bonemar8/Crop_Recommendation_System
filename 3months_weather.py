import requests
from bs4 import BeautifulSoup
import pandas as pd

# fetches high/low tempeture, rainfall of next 3 month including current one

url = "https://weather.com/tr-TR/weather/monthly/l/33d1e415eb66f3e1ab35c3add45fccf4512715d329edbd91c806a6957e123b49"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

historical_section = soup.find("div", class_="Almanac--monthlyBlock--tkGzw")
rows = historical_section.find_all("tr", class_="Almanac--row--s26zS")

months = []
high_temps = []
low_temps = []
rainfalls = []

for row in rows:
    month = row.find("td", class_="Almanac--rowLabel--KJ4rd")
    if month:
        month = month.text.strip()
    else:
        continue 

    temps = row.find_all("span", {"data-testid": "TemperatureValue"})
    if len(temps) >= 2:
        high_temp = temps[0].text.replace("°", "").strip()
        low_temp = temps[1].text.replace("°", "").strip()
    else:
        continue  

    rainfall = row.find("span", {"data-testid": "AccumulationValue"})
    rainfall = rainfall.text.strip() if rainfall else "N/A"
    
    months.append(month)
    high_temps.append(high_temp)
    low_temps.append(low_temp)
    rainfalls.append(rainfall)

weather = {
    "Month": months,
    "High Temperature": high_temps,
    "Low Temperature": low_temps,
    "Rainfall (mm)": rainfalls,
}

df = pd.DataFrame(weather)

df.to_csv("3months_weather.csv", index=False)

print("data is saved:")
print(df)
