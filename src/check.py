# Run this to see what your training Lam looks like
import cv2, os
train_lam = r"C:\Dataset\1_UAlpha40_Mendeley\static_signs\train\Lam"
img = cv2.imread(os.path.join(train_lam, os.listdir(train_lam)[0]))
cv2.imshow("Training Lam", img)
cv2.waitKey(0)