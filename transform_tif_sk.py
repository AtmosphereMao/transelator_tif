from skimage import io, feature, transform
import matplotlib.pyplot as plt
import numpy as np

# 读取输入和基准图像
input_img = io.imread('rcx.tif', as_gray=True)
reference_img = io.imread('ldk.tif', as_gray=True)

# 提取输入和基准图像中的特征点
input_keypoints = feature.blob_dog(input_img, max_sigma=30, threshold=.1)
reference_keypoints = feature.blob_dog(reference_img, max_sigma=30, threshold=.1)

# 提取特征描述符
extractor = feature.BRIEF(descriptor_size=512, patch_size=64)
extractor.extract(input_img, input_keypoints)
input_descriptors = extractor.descriptors
extractor.extract(reference_img, reference_keypoints)
reference_descriptors = extractor.descriptors

# 比较两个图像中的特征点并找到它们之间的对应
matcher = feature.match_descriptors(input_descriptors, reference_descriptors)
input_matched_keypoints = input_keypoints[matcher[:, 0]][:, ::-1]
reference_matched_keypoints = reference_keypoints[matcher[:, 1]][:, ::-1]

# 计算仿射变换矩阵
tform = transform.AffineTransform()
tform.estimate(input_matched_keypoints, reference_matched_keypoints)

# 应用仿射变换矩阵，以将图像对齐
input_transformed = transform.warp(input_img, tform, mode='edge', preserve_range=True)

# 可视化结果
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))
ax1, ax2, ax3 = ax.ravel()
ax1.imshow(reference_img, cmap='gray')
ax1.set_title('Reference image')

ax2.imshow(input_img, cmap='gray')
ax2.set_title('Input image')

ax3.imshow(input_transformed, cmap='gray')
ax3.set_title('Aligned image')
plt.show()
