import kNN
from numpy import array
import matplotlib
import matplotlib.pyplot as plt

datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')

fig = plt.figure()
ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
# ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
ax.scatter(datingDataMat[:,0], datingDataMat[:,1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
plt.show()