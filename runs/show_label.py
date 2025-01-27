import os
log_path="./"
#读取本文件下的所有文件夹,不包含文件
log_list = [x for x in os.listdir(log_path) if os.path.isdir(log_path + x)]
#排序
log_list.sort()

for log in log_list:
    file_path = os.path.join(log_path, log)
    #打开file_path下的tensorboard文件
    print(log)
    os.system("tensorboard --logdir=" + file_path)
