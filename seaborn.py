import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("zomato.csv")
print(data.head())
print(data.info())
print(data.columns)

#count plot
sns.countplot(x="listed_in(type)", data=data)

plt.title("Restaurant Types")
plt.xticks(rotation=45)
plt.show()