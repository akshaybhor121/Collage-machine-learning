import matplotlib.pyplot as plt

days = [1,2,3,4,5,6,7]
temp = [30,32,36,33,30,35,36]
plt.plot(days,temp,color="red",linewidth=2)
plt.title("Temp. of 1 week ")
plt.ylabel("Temp")
plt.xlabel("Days")
plt.show()