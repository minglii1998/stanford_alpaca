import os
import os.path as osp

dir_p = 'logs'
save_f = 'cot_test_results.csv'

result_dict = {}
for path, dir_list, file_list in os.walk(dir_p):
    for f in file_list:
        if f.endswith('.txt'):
            dir_name = path.split('/')[-1]
            if dir_name not in result_dict.keys():
                result_dict[dir_name] = {}

            f_name = f
            task_name = f_name.split('-')[0]            
            task_name = task_name[1:] # Delete special tag accordingly

            fullpath = os.path.join(path, f)
            with open(fullpath,'r') as ff:
                lines = ff.readlines()
                result_line = lines[-2]
                if 'Final Acc' not in result_line:
                    continue
                result = result_line.strip().split(':')[-1]
                result = result[0:6]
                result = str(float(result)*100)
                result = result[0:6]
                result_dict[dir_name][task_name] = result

            print('Finish:',fullpath)

        pass

# dict to list
task_list = [
    'aqua',
    'gsm8k',
    'addsub',
    'multiarith',
    'commonsensqa',
    'strategyqa',
    'svamp',
    'singleeq',
    'bigbench_date',
    'object_tracking',
    'coin_flip'
]
result_list = []
for k in result_dict.keys():
    each_raw = [k]
    for t in task_list:
        if t in result_dict[k].keys():
            each_raw.append(result_dict[k][t])
        else:
            each_raw.append(0)
    result_list.append(each_raw)
result_list = sorted(result_list)

import csv

with open('','w') as csv_f:
    writer = csv.writer(csv_f)
    writer.writerow(['']+task_list)
    writer.writerows(result_list)


pass