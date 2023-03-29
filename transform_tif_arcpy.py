# coding:utf-8
import arcpy

# 设置工作空间和文件路径
arcpy.env.workspace = "G:/features/gdal2tiles-leaflet/data"
rcx = "rcx.tif"
rcx_corrected = "rcx_corrected.tif"

# 定义控制点
ctrl_pts = arcpy.Point()
ctrl_pts.XYT = [(746539.33, 2564946.371, 0),
                 (746536.517, 2564947.03, 0)]

# 定义仿射变换类型
transformation = "SIMILARITY"

# 进行地理配准
arcpy.Warp_management(rcx, transformation,
                      "", "", "",
                      "POLYORDER1", "BILINEAR",
                      "DONT_CLIP",
                      "NO_RESAMPLING", "REGULAR",
                      ctrl_pts)

# 保存校准后的图像
arcpy.management.SaveToLayerFile(rcx, rcx_corrected, "TIFF")
