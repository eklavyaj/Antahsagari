import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

images = os.listdir('images')
images.sort()

for image in images:

	path = "images/"+image
	img = cv2.imread(path)

	# cv2.imshow("img", img)
	
	# gamma correction (change gamma to observe different results)
	gamma = 1.4
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
	dst = cv2.LUT(img, table)

	# adaptive histogram equalization using CLAHE (change cliplimit and titleGridSize to observe different results)
	dst = cv2.cvtColor(dst, cv2.COLOR_BGR2YCrCb)
	Y, Cr, Cb = cv2.split(dst)
	clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
	Y = clahe.apply(Y)
	dst = cv2.merge((Y, Cr, Cb))
	dst = cv2.cvtColor(dst, cv2.COLOR_YCrCb2BGR)
	
	# denoising the image after equalization
	dst = cv2.fastNlMeansDenoisingColored(dst,None,4,4,3,7)

	# varying contrast and brightness (change alpha and beta to observe different results)
	dst = cv2.convertScaleAbs(dst, alpha = 1.3, beta = 1.3)


	cv2.imwrite("results/"+image, dst)

	# cv2.waitKey(0)




