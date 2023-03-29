import cv2
from osgeo import gdal
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure
import argparse

# 窗口大小
g_window_wh  = [800, 600]


class struct_getPoint:
    def __init__(self, image, window):
        self.location_click = [0, 0]
        self.location_release = [0, 0]
        self.image_original = image.copy()
        self.image_show = self.image_original[0: g_window_wh[1], 0:g_window_wh[0]]
        self.location_win = [0, 0]
        self.location_win_click = [0, 0]
        self.image_zoom = self.image_original.copy()
        self.zoom = 1
        self.step = 0.1
        self.window_name = window
        self.point = []

    # OpenCV鼠标事件
    def getPoint(self):

        def mouse_callback(event, x, y, flags, param):

            def check_location(img_wh, win_wh, win_xy):
                for i in range(2):
                    if win_xy[i] < 0:
                        win_xy[i] = 0
                    elif win_xy[i] + win_wh[i] > img_wh[i] and img_wh[i] > win_wh[i]:
                        win_xy[i] = img_wh[i] - win_wh[i]
                    elif win_xy[i] + win_wh[i] > img_wh[i] and img_wh[i] < win_wh[i]:
                        win_xy[i] = 0
                # print(img_wh, win_wh, win_xy)

            # 计算缩放倍数
            # flag：鼠标滚轮上移或下移的标识, step：缩放系数，滚轮每步缩放0.1, zoom：缩放倍数
            def count_zoom(flag, step, zoom, zoom_max):
                if flag > 0:  # 滚轮上移
                    zoom += step
                    if zoom > 1 + step * 20:  # 最多只能放大到3倍
                        zoom = 1 + step * 20
                else:  # 滚轮下移
                    zoom -= step
                    if zoom < zoom_max:  # 最多只能缩小到0.1倍
                        zoom = zoom_max
                zoom = round(zoom, 2)  # 取2位有效数字
                return zoom

            if event or flags:
                w2, h2 = g_window_wh  # 窗口的宽高
                h1, w1 = param.image_zoom.shape[0:2]  # 缩放图片的宽高
                if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
                    param.location_click = [x, y]  # 左键点击时，鼠标相对于窗口的坐标
                    param.location_win_click = [param.location_win[0],
                                                param.location_win[1]]  # 窗口相对于图片的坐标，不能写成location_win = g_location_win

                elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
                    param.location_release = [x, y]  # 左键拖曳时，鼠标相对于窗口的坐标
                    if w1 <= w2 and h1 <= h2:  # 图片的宽高小于窗口宽高，无法移动
                        param.location_win = [0, 0]
                    elif w1 >= w2 and h1 < h2:  # 图片的宽度大于窗口的宽度，可左右移动
                        param.location_win[0] = param.location_win_click[0] + param.location_click[0] - \
                                                param.location_release[0]
                    elif w1 < w2 and h1 >= h2:  # 图片的高度大于窗口的高度，可上下移动
                        param.location_win[1] = param.location_win_click[1] + param.location_click[1] - \
                                                param.location_release[1]
                    else:  # 图片的宽高大于窗口宽高，可左右上下移动
                        param.location_win[0] = param.location_win_click[0] + param.location_click[0] - \
                                                param.location_release[0]
                        param.location_win[1] = param.location_win_click[1] + param.location_click[1] - \
                                                param.location_release[1]
                    check_location([w1, h1], [w2, h2], param.location_win)  # 矫正窗口在图片中的位置

                elif event == cv2.EVENT_MOUSEWHEEL:  # 滚轮
                    z = param.zoom  # 缩放前的缩放倍数，用于计算缩放后窗口在图片中的位置
                    zoom_max = g_window_wh[0] / param.image_original.shape[1]
                    param.zoom = count_zoom(flags, param.step, param.zoom, zoom_max)  # 计算缩放倍数
                    w1, h1 = [int(param.image_original.shape[1] * param.zoom),
                              int(param.image_original.shape[0] * param.zoom)]  # 缩放图片的宽高
                    param.image_zoom = cv2.resize(param.image_original, (w1, h1), interpolation=cv2.INTER_AREA)  # 图片缩放
                    param.location_win = [int((param.location_win[0] + x) * param.zoom / z - x),
                                          int((param.location_win[1] + y) * param.zoom / z - y)]  # 缩放后，窗口在图片的位置
                    check_location([w1, h1], [w2, h2], param.location_win)  # 矫正窗口在图片中的位置

                elif event == cv2.EVENT_RBUTTONDOWN:  # 右键选点
                    point_num = len(param.point)
                    [x_ori, y_ori] = [int((param.location_win[0] + x) / param.zoom),
                                      int((param.location_win[1] + y) / param.zoom)]
                    param.point.append([x_ori, y_ori])
                    cv2.circle(param.image_original, (x_ori, y_ori), 3, (255, 0, 0), thickness=-1)  # 画圆半径为3，并填充
                    cv2.putText(param.image_original, str(point_num + 1), (x_ori, y_ori), cv2.FONT_HERSHEY_PLAIN,
                                1.0, (0, 255, 0), thickness=1)  # 加入文字，位置，字体，尺度因子，颜色，粗细
                param.image_zoom = cv2.resize(param.image_original, (w1, h1), interpolation=cv2.INTER_AREA)  # 图片缩放
                param.image_show = param.image_zoom[param.location_win[1]:param.location_win[1] + h2,
                                   param.location_win[0]:param.location_win[0] + w2]  # 实际的显示图片
                cv2.imshow(param.window_name, param.image_show)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, g_window_wh[0], g_window_wh[1])
        cv2.imshow(self.window_name, self.image_show)
        cv2.setMouseCallback(self.window_name, mouse_callback, self)



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

# Begin processing with CV2. Involves histogram equalization and SIFT feature detection and matching
# img1 = norm_src_gray
# img2 = norm_target_gray

img1 = norm_src_gray
img2 = norm_target_gray

# Matching the historgrams between source and target. This can imporove the number of matches for SIFT
# img1 = exposure.match_histograms(img1, img2)
# img1 = img1.astype(np.uint8)

# 点选特征点
C1 = struct_getPoint(img1, "window1")
C1.getPoint()

C2 = struct_getPoint(img2, "window2")
C2.getPoint()

cv2.waitKey(0)  # 等待键盘点击事件来结束阻塞

kp1 = C1.point
kp2 = C2.point

if True:
    src_pts = np.array(kp1)
    dst_pts = np.array(kp2)

    M, mask = cv2.findHomography(src_pts, dst_pts)

# Applying the Homography Transform to the Bands of the SRC image
rows, cols = img1.shape
src1_warp = cv2.warpPerspective(src1_orig, M, (cols, rows))
src2_warp = cv2.warpPerspective(src2_orig, M, (cols, rows))
src3_warp = cv2.warpPerspective(src3_orig, M, (cols, rows))

# Writing the warped image bands to a new file with the same parameters as the source file
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

