import os

import cv2
import numpy as np

# 此程序用来测试单应性矩阵是否正确

img_coors = np.float32([[656, 494], [836, 485], [695, 961], [1014, 952]])
world_coors = np.float32([[3000, 4000], [3428, 4000], [3203.85, 5432.57], [3763.52, 5451.73]])
affine = cv2.getPerspectiveTransform(world_coors, img_coors)
# affine, mask = cv2.findHomography(world_coors, img_coors, cv2.RANSAC, 5.0)

# 原始图片位置
pics_path = '/media/huangluying/F/money/20230202-sociallstm/HNU_dataset/HNU_dataset/图片'
# 输出图像位置
out_path = '/media/huangluying/F/money/20230202-sociallstm/HNU_dataset/HNU_dataset/透视图'

pics = os.listdir(pics_path)
for p in sorted(pics):
    abs_path = os.path.join(pics_path, p)
    img = cv2.imread(abs_path)
    h, w, c = img.shape
    imgP = cv2.warpPerspective(img, np.linalg.inv(affine), (7400, 4800))  # 用变换矩阵 M 进行投影变换
    # imgP = cv2.warpPerspective(img, affine, (7400, 4800))  # 用变换矩阵 M 进行投影变换
    imgP = cv2.resize(imgP, (740, 480))
    cv2.imwrite(os.path.join(out_path, p), imgP)
