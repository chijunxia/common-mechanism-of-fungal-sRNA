# -*- coding: UTF-8 -*-
#将预测的靶向基因的结果文件转为pandas格式便于分析筛选
import time
import pandas as pd
file = input("Please enter the target file name:")
out = input("Please enter the out file name:")
# data = pd.DataFrame(pd.read_csv(sRNA, encoding='utf-8' ))
start = time.clock()
columns=['miRNA','target', 'score', 'mfe', 'mfe_ratio',
   'start', 'gap', 'mismatch', 'GU', 'seed_gap', 'seed_mismatch',
   'seed_GU', "miRNA_3'", 'aln', "target_5'"]
data = pd.DataFrame(columns=columns)

def write_record( record , dat):
    # ls = [record[i] for i in columns ] #新记录转为list
    newrow = pd.DataFrame([[record[i] for i in columns ]], columns=columns) #新记录转为dataframe
    # dat = dat.append(newrow,ignore_index=True)
    dat = pd.concat([dat, newrow], ignore_index=True) #新记录加入原表
    return dat

s =  open(file,"r")
lines = s.readlines()#读取全部内容
record = {}
a = 0
for line in lines:
    #print(line)
    if '#' in line:
        continue
    if '//' in line:
        a = a + 1
        #print(a)
        data = write_record(record,data)
        record ={}
        continue
    else:
        row = line.split()
        record[row[0]] = row[1]

data.to_csv(out , index=None)

end = time.clock()
print('Running dict time: %s Seconds'%(end-start))
