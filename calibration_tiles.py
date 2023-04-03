import json
import sys
import threading
import tiff_translator.tif_translator as tt

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
    print('=========================START CALIBRATION TILES=========================')

    while True:
        time.sleep(10)
        t = threading.Thread(target=calibrationTiles, name="calibrationTilesThread")
        t.start()


def calibrationTiles():
    timeStr = str(time.time())
    to_calibration_info_list = redis.get_zrange('to_calibration_info_list', -1)
    if to_calibration_info_list:
        for uniqueID in to_calibration_info_list:
            # 是否已经切片成功或正在切片，php服务器若处理不及时，则会发生重复切片情况
            isset_success_calibration = True if uniqueID in redis.get_zrange('to_calibration_success_list:',
                                                                             -1) else False
            isset_process_calibration = redis.get_values('to_calibration_process_list_info:' + uniqueID)
            if isset_success_calibration or isset_process_calibration:
                continue
            redis.set_values('to_calibration_process_list_info:' + str(uniqueID), timeStr)
            to_calibration_info = redis.get_values('to_calibration_info:' + uniqueID)
            if to_calibration_info:
                print("Handel with: " + uniqueID)
                try:

                    param = [
                        '-s', tif_path + '/ir/' + to_calibration_info['source_id'] + '/' + to_calibration_info['source_id'] + '.tif',
                        '-t', tif_path + '/visible/' + to_calibration_info['tif_code'] + '.tif',
                        '-p', 'geo',
                        '-w', tif_path + '/ir/' + to_calibration_info['source_id'] + '/' + to_calibration_info['source_id'] + '.tif',
                    ]

                    tt.translator(param,
                                  rcx_point=json.loads(to_calibration_info['rcx_point']),
                                  ldk_point=json.loads(to_calibration_info['ldk_point']))

                    # 切片任务（改为php服务器处理）
                    # tifInfo = {
                    #     'source_id': to_calibration_info['source_id'],
                    #     'type': 'ir'
                    # }
                    # cutTifMsg = hashlib.md5(timeStr).hexdigest()
                    # redis.set_values('to_tiles_info:' + cutTifMsg, tifInfo)
                    # redis.set_zrange('to_tiles_info_list:', {cutTifMsg, cutTifMsg})

                    # 完成配准
                    redis.set_zrange('to_calibration_success_list:', {uniqueID: timeStr})
                    print("Finished: " + uniqueID)
                except Exception as e:
                    info = {}
                    info['msg'] = '无法配准，请找开发人员或重新上传tif'
                    info['time'] = int(time.time())
                    info['num'] = 0
                    redis.set_values('show_web_info:' + uniqueID, info, 7200)
                    redis.del_zrange('to_calibration_info_list', uniqueID)
                    redis.remove_key('to_calibration_info:' + uniqueID)
                    print("error:", e)

            redis.remove_key('to_calibration_process_list_info:' + uniqueID)

if __name__ == '__main__':
    main()
