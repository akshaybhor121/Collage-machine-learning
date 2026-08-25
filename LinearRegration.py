import numpy as np
import matplotlib.pyplot as plt

X=[1,2,3,4,5]
Y=[3,5,6,3,8]
n=len(X)

sum_x=0
sum_y=0

for i in range(n):
    sum_x+=X[i]
    sum_y+=Y[i]

print("Sum of X:",sum_x)
print("Sum of Y:",sum_y)

mean_x=sum_x/n
mean_y=sum_y/n

print("Mean of X:",mean_x)
print("Mean of y:",mean_y)

numerator=0

for i in range(n):
    numerator+=(X[i]-mean_x) * (Y[i]-mean_y)

denominator = 0;
for i in range(n):
    denominator+=(X[i]-mean_x)**2

m=numerator/denominator
print("Slope:",m)

c=mean_y-(m*mean_x)
print("Intercept:",c)

Y_pred=[]

for i in range(n):
    predection=m*X[i]+c
    Y_pred.append(predection)

print("Result:",predection)

plt.scatter(X,Y)
plt.plot(X,Y_pred)
plt.show()








