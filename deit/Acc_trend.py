"""
localhost env: 
conda activate tSNE

usage: 
python .\Acc_trend.py .\LP_FT_logs\cifar100_10.txt

"""
import sys
import json
import matplotlib.pyplot as plt

assert len(sys.argv) == 2
file_path = sys.argv[1]
# file_path = "log.txt"
acc_1 = []

with open(file_path, 'r') as f:
    lines = f.readlines()

for line_num, data in enumerate(lines):
    # print(f"Line {line_num + 1}: {data.strip()}")  # OK, strip del '\n' in the end of `data`.
    d = json.loads(data)  # str -> dict
    acc_1.append(d['test_acc1'])  # a list of float

# print(acc_1)
plt.plot(acc_1)
max_index = acc_1.index(max(acc_1))
print(max_index, max(acc_1))
print(max_index, round(max(acc_1), 3))
plt.scatter(max_index, max(acc_1), color='red', marker='*', s=100)
# plt.text(max_index, max(acc_1)+1, str(round(max(acc_1), 3)), color='red', fontsize=10)  # for cifar
plt.text(max_index, max(acc_1)+0.3, str(round(max(acc_1), 3)), color='red', fontsize=10)  # for imagenet
plt.show()
