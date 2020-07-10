import cv2
import matplotlib.pyplot as plt
import numpy as np

x = np.zeros((255, 255))
tl = (255/3,255/3)
br = (int(255/1.5), int(255/1.5))
cv2.rectangle(x, tl, br, 255/2, -1)
# cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
# cv2.imshow('bb', x)
# plt.figure()
# plt.imshow(x)
# plt.show()

cv2.imwrite('bb.png', x)