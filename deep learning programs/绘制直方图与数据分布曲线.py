
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = np.random.rand(10000)

sns.distplot(data, bins=10)
plt.show()

data = np.random.randn(10000)
sns.distplot(data, bins=10)
plt.show()

