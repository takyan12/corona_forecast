import numpy as np 
import pandas as pd 
from fbprophet import Prophet

%matplotlib inline

df = pd.read_csv("newyork.csv")

df.head()

df['Date'] = pd.to_datetime(df['Date'])
regions = df.groupby(df.Province)

print("Total regions :", len(regions))
print("-------------")

for name, group in regions:
    print(name, " : ", len(group))

PREDICTING_FOR = "New York"
date_price = regions.get_group(PREDICTING_FOR)[['Date', 'Deaths']].reset_index(drop=True)
date_price = date_price.rename(columns={'Date':'ds', 'Deaths':'y'})
m = Prophet()
m.fit(date_price)
future = m.make_future_dataframe(periods=45)
forecast = m.predict(future)
forecast.tail()
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)
