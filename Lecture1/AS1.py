import numpy as np

a = np.full((2,3),4)
b = np.array([[1,2,3],[4,5,6]])
c = np.eye(2,3)
d = a + b + c 

print(d)

a = np.array([[1,2,3,4,5],
              [5,4,3,2,1],
              [6,7,8,9,0],
              [0,9,8,7,6]])

temp = 0

for i in range(4):
    for j in range(5):
        temp += a[i, j]

print(temp)
print(np.transpose(a))