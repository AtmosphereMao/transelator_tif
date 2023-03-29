import cv2
from osgeo import gdal
from osgeo import osr
import numpy as np
from matplotlib import pyplot as plt
import argparse

def getSRSPair(dataset):
    '''
    获得给定数据的投影参考系和地理参考系
    :param dataset: GDAL地理数据
    :return: 投影参考系和地理参考系
    '''
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs


def geo2lonlat(dataset, x, y):
    '''
    将投影坐标转为经纬度坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param x: 投影坐标x
    :param y: 投影坐标y
    :return: 投影坐标(x, y)对应的经纬度坐标(lon, lat)
    '''
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(prosrs, geosrs)
    coords = ct.TransformPoint(x, y)
    return coords[:2]


def lonlat2geo(dataset, lon, lat):
    '''
    将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param lon: 地理坐标lon经度
    :param lat: 地理坐标lat纬度
    :return: 经纬度坐标(lon, lat)对应的投影坐标
    '''
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(geosrs, prosrs)
    coords = ct.TransformPoint(lon, lat)
    return coords[:2]


def geo2imagexy(dataset, x, y):
    '''
    根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param x: 投影或地理坐标x
    :param y: 投影或地理坐标y
    :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    '''
    trans = dataset.GetGeoTransform()
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解


def plt2imagexy(dataset, lat, lon):
    geo_transform = dataset.GetGeoTransform()
    # 2. 使用matplotlib.pyplot获取坐标系对象
    extent = (geo_transform[0],
              geo_transform[0] + geo_transform[1] * dataset.RasterXSize,
              geo_transform[3] + geo_transform[5] * dataset.RasterYSize,
              geo_transform[3])
    fig, ax = plt.subplots()
    ax.imshow(dataset.ReadAsArray().astype('uint8')[0], extent=extent)

    x, y = ax.transData.transform((lon, lat))  # 将地理坐标转换为图像坐标
    return x, y


def translator(arguments=None):
    parser = argparse.ArgumentParser()

    # Parsing the input arguments
    parser.add_argument("-s", "--source", help="Source Filename - to be matched to Target")
    parser.add_argument("-t", "--target", help="Target Filename")
    parser.add_argument("-p", "--pattern", help="Type")
    parser.add_argument("-w", "--write", help="Output File")
    if arguments is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args=arguments)

    outputPath = args.write + "/src_warped.tif"  # 输出文件名

    print("SOUCE: ", args.source)
    print("TARGET: ", args.target)
    # Importing the images to be matched. SRC is the image to be mapped/corrected to TARGET
    src_filename = str(args.source)
    target_filename = str(args.target)

    # Opening the files and importing the imaging bands as layers
    src = gdal.Open(src_filename)
    print("band count: " + str(src.RasterCount))
    src1_orig = np.array(src.GetRasterBand(1).ReadAsArray())
    src2_orig = np.array(src.GetRasterBand(2).ReadAsArray())
    src3_orig = np.array(src.GetRasterBand(3).ReadAsArray())

    rcx_gt = src.GetGeoTransform()

    # del src # Closing the database

    target = gdal.Open(target_filename)
    print("band count: " + str(target.RasterCount))
    target1 = np.array(target.GetRasterBand(1).ReadAsArray())
    target2 = np.array(target.GetRasterBand(2).ReadAsArray())
    target3 = np.array(target.GetRasterBand(3).ReadAsArray())

    ldk_gt = target.GetGeoTransform()

    # del target # Closing the database

    # Normalizing to UINT8
    src1 = cv2.normalize(src1_orig, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    src2 = cv2.normalize(src2_orig, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    src3 = cv2.normalize(src3_orig, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    target1 = cv2.normalize(target1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    target2 = cv2.normalize(target2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    target3 = cv2.normalize(target3, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Stacking the satellite image bands into a single image for both SRC and TARGET
    norm_src_rgb = np.dstack((src1, src2, src3))
    norm_target_rgb = np.dstack((target1, target2, target3))

    # Normalizing and converting the images to properly handle them as color images for CV2
    norm_src_gray = cv2.cvtColor(norm_src_rgb, cv2.COLOR_BGR2GRAY)
    norm_target_gray = cv2.cvtColor(norm_target_rgb, cv2.COLOR_BGR2GRAY)

    # Begin processing with CV2. Involves histogram equalization and SIFT feature detection and matching
    img1 = norm_src_gray
    img2 = norm_target_gray

    # 读取图像文件
    rcx_img = norm_src_rgb
    ldk_img = norm_target_rgb

    # 判断尺寸是否一致
    ldk_width = target.RasterXSize
    ldk_height = target.RasterYSize
    rcx_width = src.RasterXSize
    rcx_height = src.RasterYSize

    if rcx_width != ldk_width or rcx_height != ldk_height:
        # 计算裁剪或缩放后的尺寸
        if rcx_width > ldk_width:
            width = ldk_width
            height = int(float(rcx_height * ldk_width) / rcx_width)
        else:
            height = ldk_height
            width = int(float(rcx_width * ldk_height) / rcx_height)

        # 裁剪或缩放图像
        # rcx_img = cv2.imread('rcx.tif', cv2.IMREAD_UNCHANGED)
        # rcx_img = cv2.resize(rcx_img, (width, height))
        # img1 = cv2.resize(img1, (width, height))

        # rcx_geotransform = rcx_gt
        # ldk_geotransform = ldk_gt

        # 更新元数据信息
        # rcx_width, rcx_height = width, height
        # rcx_geotransform = (rcx_geotransform[0], ldk_geotransform[1], rcx_geotransform[2],
        #                     rcx_geotransform[3], ldk_geotransform[5], rcx_geotransform[5])
        # rcx_projection = ldk_projection

    # geo地理坐标转图坐标
    if args.pattern == "geo":
        rcx_pts = []
        ldk_pts = []

        # 投影坐标
        rcx_point = [[746465.516, 2565001.615],
                     [746465.643, 2564996.498],
                     [746478.733, 2565001.912],
                     [746478.627, 2564996.805],
                     [746470.429, 2564998.404],
                     [746470.398, 2564999.865],
                     [746475.219, 2565000.378],
                     [746475.251, 2564998.494],
                     [746478.870, 2564999.326],
                     [746478.659, 2564997.506],
                     [746478.796, 2565001.178],
                     [746465.547, 2565000.185],
                     [746465.571, 2564998.317],
                     [746478.631, 2564998.254],
                     [746478.811, 2565000.063]]

        ldk_point = [[746524.527, 2564945.957],
                     [746524.804, 2564932.463],
                     [746559.650, 2564946.751],
                     [746559.293, 2564933.296],
                     [746537.535, 2564937.480],
                     [746537.436, 2564941.349],
                     [746550.185, 2564942.745],
                     [746550.020, 2564937.784],
                     [746559.834, 2564940.000],
                     [746559.338, 2564935.171],
                     [746559.669, 2564944.862],
                     [746524.624, 2564942.159],
                     [746524.703, 2564937.265],
                     [746559.760, 2564941.895],
                     [746559.231, 2564937.106]]

        countPoint = len(rcx_point)

        for i in range(countPoint):
            rcx_lat, rcx_lon, *_ = [float(x) for x in rcx_point[i]]
            ldk_lat, ldk_lon, *_ = [float(x) for x in ldk_point[i]]

            rcx_lon -= 0.1

            # rcx_lat, rcx_lon = geo2lonlat(src, rcx_lat, rcx_lon)  # 转经纬度
            # ldk_lat, ldk_lon = geo2lonlat(target, ldk_lat, ldk_lon)

            rcx_px, rcx_py = geo2imagexy(target, rcx_lat, rcx_lon)  # 以源图进行变换
            ldk_px, ldk_py = geo2imagexy(src, ldk_lat, ldk_lon)

            # rcx_px, rcx_py = gdal.ApplyGeoTransform(rcx_gt, rcx_lon, rcx_lat)  # 转投影坐标
            # ldk_px, ldk_py = gdal.ApplyGeoTransform(ldk_gt, ldk_lon, ldk_lat)

            rcx_pts.append((rcx_px, rcx_py))
            ldk_pts.append((ldk_px, ldk_py))

    else:
        print("Please click on the corresponding points on the images.")
        # 显示两幅图像
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].imshow(rcx_img)
        ax[0].set_title('RCX Image')

        ax[1].imshow(ldk_img)
        ax[1].set_title('LDK Image')

        # 手动选择控制点
        plt.subplots_adjust(bottom=0.15)
        rcx_pts = plt.ginput(n=4, timeout=0)
        ldk_pts = plt.ginput(n=4, timeout=0)

    # print(rcx_pts)
    # print(ldk_pts)

    # 转换控制点为 NumPy 数组
    rcx_pts = np.array(rcx_pts)
    ldk_pts = np.array(ldk_pts)

    # 将控制点转换为单精度浮点数
    src_pts = np.float32([i for i in rcx_pts]).reshape(-1, 1, 2)
    dst_pts = np.float32([i for i in ldk_pts]).reshape(-1, 1, 2)

    if True:
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

    # Removing the scaling factor from the homography matrix
    M[0, 0] = 1.0
    M[1, 1] = 1.0

    # Running some processing on the matching points to further select the best points
    bool_list = list(map(bool, matchesMask))
    src_points = src_pts[bool_list]
    dst_points = dst_pts[bool_list]
    sp = np.asarray(src_points)
    dp = np.asarray(dst_points)

    sp_y = sp[:, 0, 0]
    sp_x = sp[:, 0, 1]
    dp_y = dp[:, 0, 0]
    dp_x = dp[:, 0, 1]

    # height, width of both source and target
    height_sp = img1.shape[0]
    width_sp = img1.shape[1]
    height_dp = img2.shape[0]
    width_dp = img2.shape[1]

    # Normalizing to get the amount of pixel shift needed for the gdal geotransform
    sp_y_norm = sp_y / height_sp
    sp_x_norm = sp_x / width_sp
    dp_y_norm = dp_y / height_dp
    dp_x_norm = dp_x / width_dp

    x_shift_percent = np.subtract(sp_x_norm, dp_x_norm)
    y_shift_percent = np.subtract(sp_y_norm, dp_y_norm)
    # print(x_shift_percent)
    # x_shift_percent = list(filter(lambda num: ((num <= 0.01) and (num >= -0.01)), x_shift_percent))
    # y_shift_percent = list(filter(lambda num: ((num <= 0.01) and (num >= -0.01)), y_shift_percent))
    # print("test", x_shift_percent)
    x_pixel_shift = np.median(np.array(x_shift_percent))
    y_pixel_shift = np.median(np.array(y_shift_percent))
    # x_shift_percent_mean = x_pixel_shift
    # y_shift_percent_mean = y_pixel_shift
    # x_shift_std = np.subtract(sp_x_norm, dp_x_norm).std()
    # y_shift_std = np.subtract(sp_y_norm, dp_y_norm).std()

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

        with rasterio.open(outputPath, 'w', **kwds) as dst_dataset:
            dst_dataset.write(src1_warp, 1)
            dst_dataset.write(src2_warp, 2)
            dst_dataset.write(src3_warp, 3)

    # open dataset with update permission to update the coordinates to reflect the pixel shift/translation
    ds = gdal.Open(outputPath, gdal.GA_Update)
    # get the geotransform as a tuple of 6
    gt = ds.GetGeoTransform()
    # unpack geotransform into variables
    x_tl, x_res, dx_dy, y_tl, dy_dx, y_res = gt

    # compute shift of N pixel(s) in X direction
    shift_x = -x_pixel_shift * x_res
    # compute shift of M pixels in Y direction
    # y_res likely negative, because Y decreases with increasing Y index
    shift_y = (-y_pixel_shift / 2) * y_res

    # make new geotransform
    gt_update = (x_tl + shift_x, x_res, dx_dy, y_tl + shift_y, dy_dx, y_res)
    # assign new geotransform to raster
    ds.SetGeoTransform(gt_update)
    # ensure changes are committed
    ds.FlushCache()
    ds = None


if __name__ == '__main__':
    param = [
        '-s', 'rcx.tif',
        '-t', 'ldk.tif',
        '-p', 'geo',
        '-w', './data',
    ]
    translator(param)
