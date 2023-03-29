import cv2
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import argparse

parser = argparse.ArgumentParser()

# Parsing the input arguments
parser.add_argument("-s", "--source", help="Source Filename - to be matched to Target")
parser.add_argument("-t", "--target", help="Target Filename")

args = parser.parse_args()
print("SOUCE: ", args.source)
print("TARGET: ", args.target)

# Importing the images to be matched. SRC is the image to be mapped/corrected to TARGET
src_filename = str(args.source)
target_filename = str(args.target)

# Opening the files and importing the imaging bands as layers
src = gdal.Open(src_filename)
print ("band count: " + str(src.RasterCount))
src1_orig = np.array(src.GetRasterBand(1).ReadAsArray())
src2_orig = np.array(src.GetRasterBand(2).ReadAsArray())
src3_orig = np.array(src.GetRasterBand(3).ReadAsArray())

del src # Closing the database

target = gdal.Open(target_filename)
print ("band count: " + str(target.RasterCount))
target1 = np.array(target.GetRasterBand(1).ReadAsArray())
target2 = np.array(target.GetRasterBand(2).ReadAsArray())
target3 = np.array(target.GetRasterBand(3).ReadAsArray())

del target # Closing the database


# Normalizing to UINT8
src1 = cv2.normalize(src1_orig, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
src2 = cv2.normalize(src2_orig, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
src3 = cv2.normalize(src3_orig, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

target1 = cv2.normalize(target1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
target2 = cv2.normalize(target2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
target3 = cv2.normalize(target3, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


# Stacking the satellite image bands into a single image for both SRC and TARGET
norm_src_rgb = np.dstack((src1,src2, src3))
norm_target_rgb = np.dstack((target1,target2, target3))

# Normalizing and converting the images to properly handle them as color images for CV2
norm_src_gray = cv2.cvtColor(norm_src_rgb, cv2.COLOR_BGR2GRAY)
norm_target_gray = cv2.cvtColor(norm_target_rgb, cv2.COLOR_BGR2GRAY)

# 读取图像文件
rcx_img = norm_src_rgb
ldk_img = norm_target_rgb

# 显示两幅图像
fig, ax = plt.subplots(1, 2, figsize=(20, 20))
ax[0].imshow(rcx_img)
ax[0].set_title('RCX Image')

ax[1].imshow(ldk_img)
ax[1].set_title('LDK Image')

# 手动选择控制点
plt.subplots_adjust(bottom=0.15)
rcx_pts = plt.ginput(n=6)
ldk_pts = plt.ginput(n=6)

# 转换控制点为 NumPy 数组
rcx_pts = np.array(rcx_pts)
ldk_pts = np.array(ldk_pts)

# 将控制点转换为单精度浮点数
rcx_pts = rcx_pts.astype(np.float32)
ldk_pts = ldk_pts.astype(np.float32)

# 进行 Homography 变换
if len(rcx_pts) >= 4 and len(ldk_pts) >= 4:
    M, mask = cv2.findHomography(rcx_pts, ldk_pts, cv2.RANSAC)

    # # 调整 RCX 图像的大小以适应 LDK 图像
    rows, cols = ldk_img.shape[:2]
    src1_warp = cv2.warpPerspective(src1_orig, M, (cols, rows))
    src2_warp = cv2.warpPerspective(src2_orig, M, (cols, rows))
    src3_warp = cv2.warpPerspective(src3_orig, M, (cols, rows))

    import rasterio

    with rasterio.open(src_filename) as src_dataset:

        # Get a copy of the source dataset's profile. Thus our
        # destination dataset will have the same dimensions,
        # number of bands, data type, and georeferencing as the
        # source dataset.
        kwds = src_dataset.profile

        # Change the format driver for the destination dataset to
        # 'GTiff', short for GeoTIFF.
        kwds['driver'] = 'GTiff'

        with rasterio.open('src_warped.tif', 'w', **kwds) as dst_dataset:
            dst_dataset.write(src1_warp, 1)
            dst_dataset.write(src2_warp, 2)
            dst_dataset.write(src3_warp, 3)

    # # open dataset with update permission to update the coordinates to reflect the pixel shift/translation
    # ds = gdal.Open('src_warped.tif', gdal.GA_Update)
    # # get the geotransform as a tuple of 6
    # gt = ds.GetGeoTransform()
    # # unpack geotransform into variables
    # x_tl, x_res, dx_dy, y_tl, dy_dx, y_res = gt
    #
    # # compute shift of N pixel(s) in X direction
    # shift_x = -x_pixel_shift * x_res
    # # compute shift of M pixels in Y direction
    # # y_res likely negative, because Y decreases with increasing Y index
    # shift_y = (-y_pixel_shift / 2) * y_res
    #
    # # make new geotransform
    # gt_update = (x_tl + shift_x, x_res, dx_dy, y_tl + shift_y, dy_dx, y_res)
    # # assign new geotransform to raster
    # ds.SetGeoTransform(gt_update)
    # # ensure changes are committed
    # ds.FlushCache()
    # ds = None

    #
    # # 保存配准后的图像
    # cv2.imwrite('rcx_warped.tif', rcx_warped)
    #
    # # 显示配准后的图像
    # fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    # ax[0].imshow(ldk_img)
    # ax[0].set_title('LDK Image')
    # ax[1].imshow(rcx_warped)
    # ax[1].set_title('RCX Warped Image')
    # plt.show()
else:
    print("Error: The input arrays should have at least 4 corresponding point sets to calculate Homography in function 'cv::findHomography'")
