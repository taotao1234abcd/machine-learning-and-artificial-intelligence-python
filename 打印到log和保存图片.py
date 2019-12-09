
import time
import matplotlib.pyplot as plt

import logging
LOG_FORMAT = "%(asctime)s - %(message)s"
logging.basicConfig(filename=time.strftime("%Y-%m-%d %H%M%S",time.localtime(time.time()))+'.log',
	level=logging.INFO, format=LOG_FORMAT)
# logging.info("haha")


plt.plot(train_accuracy_list, lw=1)
plt.savefig(time.strftime("%Y-%m-%d %H%M%S", time.localtime(time.time())) + '.png', dpi=200)
plt.show()

# print('Epoch:%3d, train accuracy: %.4f, test accuracy: %.4f' % (epoch + 1, accuracy_train, accuracy_test))
logging.info('Epoch:%3d, train accuracy: %.4f, test accuracy: %.4f' % (epoch + 1, accuracy_train, accuracy_test))

