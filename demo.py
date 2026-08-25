import pandas as pd

file_name = input("enter file Name")
data = pd.read_csv(file_name)

print(data)