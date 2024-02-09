import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist


(train_images, train_labels), (_, _) = mnist.load_data()

indices_1 = np.where(train_labels == 1)[0]
indices_3 = np.where(train_labels == 3)[0]
indices_7 = np.where(train_labels == 7)[0]

plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(train_images[indices_1[np.random.randint(0, len(indices_1))]], cmap='gray')
plt.title('Number 1')

plt.subplot(1, 3, 2)
plt.imshow(train_images[indices_3[np.random.randint(0, len(indices_3))]], cmap='gray')
plt.title('Number 3')

plt.subplot(1, 3, 3)
plt.imshow(train_images[indices_7[np.random.randint(0, len(indices_7))]], cmap='gray')
plt.title('Number 7')

plt.tight_layout()
plt.show()