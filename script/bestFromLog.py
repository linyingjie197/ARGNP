import re 

# 打开文件
with open('CIFAR10.log', 'r') as file:
 # 逐行读取
    Mdict = {}
    flag = 0
    for line in file:
        if line.find('** repeat 4') != -1:
            flag = 1 
        if line.find('test  result') != -1 and flag == 1:
            match = re.search(r'\[\s*(\d+)\s*\].*metric\s*:\s*(\d+\.\d+)', line)
            if match:
                index = match.group(1)
                metric = match.group(2)
                #print(f'Index: {index}, Metric: {metric}')
                Mdict[index] = metric
    
sorted_metric = sorted(Mdict.items(), key=lambda x: float(x[1]), reverse=True)
print("\nsorted_metric:")
output_lines = []
for index, metric in sorted_metric:
    output_lines.append(f'Index: {index:<10} Metric: {metric}')
output = '\n'.join(output_lines)
print(output)