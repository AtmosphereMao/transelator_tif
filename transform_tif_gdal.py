import cv2
import numpy as np

# def select_roi(img):
#     roi = cv2.selectROI(img)
#     return roi
#
# def detect_features(img, roi):
#     sift = cv2.xfeatures2d.SIFT_create()
#     kp1, des1 = sift.detectAndCompute(rcx, None)
#
#     x, y, w, h = roi
#     roi_img = img[y:y+h, x:x+w]
#
#     kp2, des2 = sift.detectAndCompute(roi_img, None)
#
#     FLANN_INDEX_KDTREE = 1
#     index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#     search_params = dict(checks=50)
#     flann = cv2.FlannBasedMatcher(index_params, search_params)
#     matches = flann.knnMatch(des1, des2, k=2)
#
#     good = []
#     for m, n in matches:
#         if m.distance < 0.7 * n.distance:
#             good.append(m)
#
#     src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#     dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
#
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#
#     h, w = img.shape
#     pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
#
#     dst = cv2.perspectiveTransform(pts, M)
#
#     ldk_with_kp = cv2.drawKeypoints(img, kp2, None, color=(0, 255, 0), flags=0)
#     final_img = cv2.polylines(ldk_with_kp, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
#
#     cv2.imshow('Final', final_img)
#     cv2.imwrite('rcx_corrected_1.tif', final_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# rcx = cv2.imread('rcx.tif', 0)
# ldk = cv2.imread('ldk.tif', 0)
# x, y = ldk.shape[0:2]
# ldk = cv2.resize(ldk, (int(y/ 16), int(x/ 16)))
#
# roi = select_roi(ldk)
# detect_features(ldk, roi)

# load images
rcx = cv2.imread('rcx.tif', 0)
ldk = cv2.imread('ldk.tif', 0)

# resize ldk image
ldk_resized = cv2.resize(ldk, (0,0), fx=0.1, fy=0.1)
rcx = cv2.resize(rcx, None, fx=0.2, fy=0.2)
# ldk_resized = cv2.applyColorMap(ldk_resized, cv2.COLORMAP_JET)
# ldk_resized = cv2.equalizeHist(ldk_resized)
# 定义膨胀核的大小和迭代次数
kernel_size = (5, 5)
iterations = 1

# 创建膨胀核
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

# 对图像进行膨胀处理
rcx_dilation = cv2.dilate(rcx, kernel, iterations)

# # 用原始图像减去膨胀后的图像，得到边缘
# rcx_edge = rcx - rcx_dilation
#
# # 加粗边缘
# edge_size = 2
# kernel_size = (edge_size, edge_size)
# kernel = np.ones(kernel_size, np.uint8)
# rcx_edge = cv2.erode(rcx_edge, kernel, iterations)

# 将边缘添加回原始图像中
# rcx = rcx + rcx_edge
#
# rcx = cv2.resize(rcx, (0,0), fx=0.5, fy=0.5)
#
# # 使用Otsu阈值法进行二值化
# ret, rcx = cv2.threshold(rcx, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

roi_x, roi_y, roi_w, roi_h = cv2.selectROI(ldk_resized)

ldk_roi = ldk_resized[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

roi_x, roi_y, roi_w, roi_h = cv2.selectROI(rcx)

rcx = rcx[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]


# define SIFT feature extractor
sift = cv2.xfeatures2d.SIFT_create()

# find keypoints and descriptors
kp_roi, des_roi = sift.detectAndCompute(ldk_roi, None)
kp_rcx, des_rcx = sift.detectAndCompute(rcx, None)

# create BFMatcher object
bf = cv2.BFMatcher()

# match descriptors
matches = bf.match(des_roi, des_rcx)

# sort matches by distance
matches = sorted(matches, key = lambda x:x.distance)

# draw matches on images
ldk_roi_matches = cv2.drawMatches(ldk_roi, kp_roi, rcx, kp_rcx, matches[:25], None, flags=2)

# calculate transformation matrix
src_pts = np.float32([kp_roi[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
dst_pts = np.float32([kp_rcx[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

# apply transformation matrix to rcx image
# rcx_aligned = cv2.warpPerspective(rcx, M, (ldk.shape[1], ldk.shape[0]))

# show images
cv2.imshow('ldk_roi_matches', ldk_roi_matches)
# cv2.imshow('rcx_aligned', rcx_aligned)
cv2.waitKey(0)
cv2.destroyAllWindows()

