import json

file_ori = ''
file_para = ''

with open(file_ori, "r") as f:
    data_ori = json.load(f)
with open(file_para, "r") as f:
    data_para = json.load(f)

para_instruct = False
para_input = True

new_data = data_ori + data_para

print('Ori data len \n',len(data_ori))
print('New data len \n',len(new_data))

with open("", "w") as fw:
    json.dump(new_data, fw, indent=4)

pass
