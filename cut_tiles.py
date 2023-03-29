import gdal2tilesG as gt
import sys
import threading

try:
    from osgeo import gdal
    from osgeo import osr
except:
    import gdal

    print('You are using "old gen" bindings. gdal2tiles needs "new gen" bindings.')
    sys.exit(1)

from redis_tools import RedisTools
import time

redis = RedisTools('192.168.31.8', 6379, 'anlly12345', 4)

tif_path = '/home/tif_data'
tif_slice_path = '/home/tif_slice_data'


def main():
    print('=========================START CUT TILES=========================')

    while True:
        time.sleep(10)
        t = threading.Thread(target=cutTiles, name="cutTilesThread")
        t.start()

def cutTiles():
    timeStr = str(time.time())
    to_tiles_info_list = redis.get_zrange('to_tiles_info_list', -1)
    if to_tiles_info_list:
        for insertID in to_tiles_info_list:
            # 是否已经切片成功或正在切片，php服务器若处理不及时，则会发生重复切片情况
            isset_success_tiles = True if insertID in redis.get_zrange('to_tiles_success_list:', -1) else False
            isset_process_tiles = redis.get_values('to_tiles_process_list_info:' + insertID)
            if isset_success_tiles or isset_process_tiles:
                continue
            redis.set_values('to_tiles_process_list_info:' + str(insertID), timeStr)
            to_tiles_info = redis.get_values('to_tiles_info:' + insertID)
            if to_tiles_info:
                try:
                    param = [
                        "none",
                        "-p", "mercator",
                        "-z", "0-24",
                    ]
                    # 可见光
                    if to_tiles_info['type'] == 'visible':
                        ldk_path = tif_path + '/visible/' + to_tiles_info['tif_code'] + '.tif'  # 切片图
                        slice_path = tif_slice_path + '/' + insertID + '/visible'  # 输出路径
                        param.append(ldk_path)
                        param.append(slice_path)
                    # 热成像
                    elif to_tiles_info['type'] == 'ir':
                        rcx_path = tif_path + '/ir/' + to_tiles_info['source_id'] + '/' + to_tiles_info['source_id'] + '.tif'
                        slice_path = tif_path + '/ir/' + to_tiles_info['source_id'] + '/'  # 输出路径
                        param.append(rcx_path)
                        param.append(slice_path)

                    gdal2tiles = gt.GDAL2Tiles(param[1:])
                    gdal2tiles.process()

                    redis.set_zrange('to_tiles_success_list:', {insertID: timeStr})

                except Exception as e:
                    info = {}
                    info['msg'] = '无法切片，请找开发人员或重新上传tif'
                    info['time'] = int(time.time())
                    info['num'] = 0
                    redis.set_values('show_web_info:' + insertID, info, 7200)
                    print("error:", e)

            redis.remove_key('to_tiles_process_list_info:' + insertID)


if __name__ == '__main__':
    main()
