import cv2
import os
import numpy

lr_path = "../datasets/ntire20/Corrupted-va-x/"
hr_path = "../datasets/ntire20/Corrupted-va-y/"

files = os.listdir(lr_path)

for name in files:
    lr_img = cv2.imread(lr_path + name)
    hr_img = cv2.imread(hr_path + name)
    print(name)
    os.remove(lr_path + name)
    os.remove(hr_path + name)
    lr_mid = lr_img.shape[1] // 2
    hr_mid = lr_mid * 4
    # crop lr img
    cv2.imwrite(lr_path + name.replace(".", "_0."), lr_img[:, :lr_mid, :])
    cv2.imwrite(lr_path + name.replace(".", "_1."), lr_img[:, lr_mid:, :])
    # crop hr img
    cv2.imwrite(hr_path + name.replace(".", "_0."), hr_img[:, :hr_mid, :])
    cv2.imwrite(hr_path + name.replace(".", "_1."), hr_img[:, hr_mid:, :])



