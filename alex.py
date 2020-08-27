#importing packages
import numpy as np

from alexnet import Alexnet

#input
np.random.seed(0)

x=np.random.uniform(size=(3,227,227))

emp=Alexnet.function(x)
print(emp)


