import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("zomato.csv")

sns.countplot(x="listed_in(type)", data=data)

plt.xticks(rotation=45)
plt.show()