from tensorboard.backend.event_processing import event_accumulator
from tensorboardX import SummaryWriter
import os

def smooth(scalars, weight):
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed



def get_log_path(log_path,env,seed,is_new):
    #列出log_path下的所有文件夹，不包含文件
    all_log_path = [x for x in os.listdir(log_path) if os.path.isdir(log_path + x)]
    if is_new:
        isisnew="new"
    else:
        isisnew="old"
    #找到既包含env又包含seed又包含isisnew的文件夹
    for log in all_log_path:
        if env in log and seed in log and isisnew in log:
            return os.path.join(log_path,log)
    return None
    
def read_data(log_file_path):
    ea = event_accumulator.EventAccumulator(log_file_path)
    ea.Reload()
    return ea
def smooth_save_data(ea,tag,smoothed_log_dir):
    if tag in ea.Tags()['scalars']:
        scalar_events = ea.Scalars(tag)
        steps = [event.step for event in scalar_events]
        values = [event.value for event in scalar_events]
        # 平滑数据
        # smoothed_values = smooth(values, 0.9)
       
        # for i in range(1, 11):
        #     steps.insert(0, 11-i)
        #     smoothed_values.insert(0, 0)
        smoothed_values = values

        
        
        
        if not os.path.exists(smoothed_log_dir):
            os.makedirs(smoothed_log_dir)
        writer = SummaryWriter(smoothed_log_dir)
        
        for step, value in zip(steps, smoothed_values):
            writer.add_scalar(tag, value, step)
        
        writer.close()
        print(f"Tag '{tag}' saved.")
    else:
        print(f"Tag '{tag}' not found.")

seeds=["1024","314159","4649","5840","42","2986"]
envs=["BlockedUnlockPickup","Unlock","UnlockPickup","RedBlueDoor"]
log_path = "./"
# save_path="../smoothed_logs"
save_path="../ori_logs"

for seed in seeds:
    for env in envs:
        for is_new in [True,False]:
            isisnew = "new" if is_new else "old"
            file_path = get_log_path(log_path,env,seed,is_new)

            ea = read_data(file_path)
            this_save_path = os.path.join(save_path,env,isisnew,seed)
            print(f"Processing {file_path} seed {seed} save to {this_save_path}")
            # exit()
            
            smooth_save_data(ea,"success_rate",this_save_path)





