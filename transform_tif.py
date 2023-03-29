import cv2
import numpy as np

# 读取热成像tif图像
rcx_img = cv2.imread('rcx.tif', cv2.IMREAD_GRAYSCALE)

# 读取可见光tif图像
ldk_img = cv2.imread('ldk.tif')
ldk_img_gry = cv2.cvtColor(ldk_img, cv2.COLOR_BGR2GRAY)

img = cv2.imread('rcx.tif')
img_gry = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

# 创建ORB特征提取器
orb = cv2.ORB_create()

# 检测关键点和生成描述符
kp1, des1 = orb.detectAndCompute(rcx_img, None)
kp2, des2 = orb.detectAndCompute(ldk_img, None)

for keypoint,descriptor in zip(kp1, des1):
    print("keypoint:", keypoint.angle, keypoint.class_id, keypoint.octave, keypoint.pt, keypoint.response, keypoint.size)
    print("descriptor: ", descriptor.shape)

for keypoint,descriptor in zip(kp2, des2):
    print("keypoint:", keypoint.angle, keypoint.class_id, keypoint.octave, keypoint.pt, keypoint.response, keypoint.size)
    print("descriptor: ", descriptor.shape)


img = cv2.drawKeypoints(image=img_gry, outImage=img, keypoints=kp1,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))


img2 = cv2.drawKeypoints(image=ldk_img_gry, outImage=ldk_img, keypoints=kp2,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))


cv2.namedWindow("img_gray",cv2.WINDOW_NORMAL)
cv2.namedWindow("new_img",cv2.WINDOW_NORMAL)
cv2.resizeWindow("img_gray", 800,600)
cv2.resizeWindow("new_img", 800,600)

cv2.imshow("img_gray", img_gry)
cv2.imshow("new_img", img)

cv2.namedWindow("ldk_img_gray",cv2.WINDOW_NORMAL)
cv2.namedWindow("ldk_new_img",cv2.WINDOW_NORMAL)
cv2.resizeWindow("ldk_img_gray", 800,600)
cv2.resizeWindow("ldk_new_img", 800,600)

cv2.imshow("ldk_img_gray", ldk_img_gry)
cv2.imshow("ldk_new_img", ldk_img)

cv2.waitKey(0)
cv2.destroyAllWindows()


# 匹配描述符
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# 根据匹配结果筛选出较好的匹配关键点对
matches = sorted(matches, key=lambda x: x.distance)
good = matches[:30]

# 获取匹配关键点对的坐标
pts1 = [kp1[m.queryIdx].pt for m in good]
pts2 = [kp2[m.trainIdx].pt for m in good]

# 计算透视变换矩阵，将热成像图像对准可见光图像
H, _ = cv2.findHomography(np.array(pts1), np.array(pts2), cv2.RANSAC, 5.0)
registered_img = cv2.warpPerspective(rcx_img, H, (ldk_img.shape[1],ldk_img.shape[0]))

# 保存对齐后的热成像图像
cv2.imwrite('registered_rcx.tif', registered_img)
