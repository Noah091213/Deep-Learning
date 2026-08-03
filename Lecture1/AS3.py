import matplotlib.pyplot as plt
import numpy as np

# Initial data
a = np.array([1,1,2,3,5,8,13,21,34])
b = np.array([1,8,28,56,70,56,28,8,1])

# Init the plot
fig, mainPlot = plt.subplots()

# Add the data, individual datasets are added individually
mainPlot.plot(a, label = 'Training accuracy')
mainPlot.plot(b, label = 'Validation accuracy')

# Make the plot look pretty
mainPlot.set_xlabel('Epocs')
mainPlot.set_ylabel('Accuracy')
mainPlot.set_title('Training and validation accuracy')
mainPlot.legend()
plt.show()