import numpy as np
import matplotlib.pyplot as plt

a = np.array([1,1,2,3,5,8,13,21,34])
b = np.array([1,8,28,56,70,56,28,8,1])

plt.plot(a, label="Training accuracy")
plt.plot(b, label="Validation accuracy")

plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("Training and validation accuracy")

plt.legend()
plt.show()