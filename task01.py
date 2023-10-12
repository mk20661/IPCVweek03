import numpy as np
import cv2
import matplotlib.pyplot as plt 


def sobel(input):
    xDrection = np.array([[-1,0,1],
                          [-2,0,2],
                          [-1,0,1]])
    
    yDrection = np.array([[-1,-2,-1],
                          [0,0,0],
                          [1,2,1]])
    
    gradient_x = np.abs(np.convolve(input, xDrection, mode='same'))
    gradient_y = np.abs(np.convolve(input, yDrection, mode='same'))
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return gradient_x, gradient_y, gradient_magnitude, gradient_direction

image = cv2.imread("coins1.png")
gradient_x, gradient_y, gradient_magnitude, gradient_direction = sobel(image)

plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(gradient_x, cmap='gray')
plt.title('x方向导数')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(gradient_y, cmap='gray')
plt.title('y方向导数')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('梯度强度')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(gradient_direction, cmap='jet')
plt.title('梯度方向')
plt.axis('off')

plt.show()


