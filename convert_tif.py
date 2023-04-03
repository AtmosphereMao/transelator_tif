import os
import time
from tqdm import tqdm
from pyodm import Node, exceptions

image_dir = './tif_assert/visibleLight/'
result_path = image_dir.split('/')[0]+'_results'
# ip = '192.168.31.8'    # 改为自己的ip
ip = 'localhost'
port = 3000

start = time.time()
images_name = os.listdir(image_dir)
images_name = [image_dir+image_name for image_name in images_name]
print(images_name)


node = Node(ip, port)

try:
    # Start a task
    print("Uploading images...")
    task = node.create_task(images_name)
    print(task.info())

    try:
        pbar = tqdm(total=100)
        processing = 0
        while True:
            info = task.info()
            if info.progress==100:
                break
            pbar.update(info.progress-processing)
            processing = info.progress
            if info.last_error!='':
                print("error ", info.last_error)

        print("{}张图消耗{}秒".format(len(images_name), time.time() - start))

        # This will block until the task is finished
        # or will raise an exception
        task.wait_for_completion()

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
