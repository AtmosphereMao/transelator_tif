import os
import time
from tqdm import tqdm
from pyodm import Node, exceptions

image_dir = './tif_assert/visibleLight/'
result_path = image_dir.split('/')[0]+'_results'
ip = '192.168.31.8'    # 改为自己的ip
# ip = 'localhost'
port = 3001

start = time.time()
images_name = os.listdir(image_dir)
images_name = [image_dir+image_name for image_name in images_name]
print(images_name)


node = Node(ip, port)

try:
    # Start a task
    print("Uploading images...")
    task = node.create_task(images_name, {'feature-type': 'sift',
                                          'feature-quality': 'high',
                                          'matcher-type': 'flann',
                                          'skip-3dmodel': True,
                                          'skip-report': True,
                                          'use-exif': True,
                                          'force-gps': True,
                                          'pc-ept': False,
                                          'cog': False,
                                          'gltf': False,
                                          'pc-quality': 'ultra',
                                          'orthophoto-compression': 'NONE',
                                          # 'orthophoto-resolution': 30,
                                          'auto-boundary': True,
                                          'crop': 0,
                                          'gps-accuracy': 1,
                                          'matcher-neighbors': 8,
                                          'max-concurrency': 8,
                                          'min-num-features': 120000,
                                          'resize-to': -1,
                                          'fast-orthophoto': True,
                                          'mesh-octree-depth': 14,
                                          'ignore-gsd': True,
                                          # 'sm-no-align': True,
                                          'dsm': True
                                          })
    print(task.info())

    try:
        print("{}张图加载消耗{}秒".format(len(images_name), time.time() - start))

        print("任务进行中，可前往 %s 进行查看" % (ip+":"+str(port)))

        # This will block until the task is finished
        # or will raise an exception
        task.wait_for_completion()

        print("{}张图完成消耗{}秒".format(len(images_name), time.time() - start))

        print("Task completed, downloading results...")

        # Retrieve results
        task.download_assets(result_path)

        print("Assets saved in ./results (%s)" % os.listdir("./results"))

        # Restart task and this time compute dtm
        task.restart({'dtm': True})
        task.wait_for_completion()

        print("Task completed, downloading results...")

        task.download_assets("./results_with_dtm")

        print("Assets saved in ./results_with_dtm (%s)" % os.listdir("./results_with_dtm"))
    except exceptions.TaskFailedError as e:
        print("\n".join(task.output()))

except exceptions.NodeConnectionError as e:
    print("Cannot connect: %s" % e)
except exceptions.NodeResponseError as e:
    print("Error: %s" % e)
