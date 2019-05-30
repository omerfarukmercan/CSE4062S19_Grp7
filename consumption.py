import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

result = pd.read_excel('calisma.xlsx')
#print(result.head())
#print(result.dtypes)
#print("----------------------")

print(result.Column1)
print(result.MT_002)

df = pd.DataFrame({
    'x': result.Column1,
    'y': result.MT_002
})

# region Description
result['date'] = pd.to_datetime(result['Column1'])
data = result.loc[:, ['MT_002']]
data = data.set_index(result.date)
data['MT_002'] = pd.to_numeric(data['MT_002'], downcast='float', errors='coerce')
#print(data)
plt.plot(data)
plt.show()

weekly = data.resample('W').sum()
plt.plot(weekly)
plt.show()

daily = data.resample('D').sum()
plt.plot(daily)
plt.show()

monthly = data.resample('M').sum()
plt.plot(monthly)
plt.show()
# endregion



print(df)


