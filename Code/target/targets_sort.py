import pandas as pd
file1 = input("Please enter the sRNA file1 name:")
data1 = pd.DataFrame(pd.read_csv(file1,  encoding='utf-8'))
file2 = input("Please enter the target file2 name:")
data2 = pd.DataFrame(pd.read_csv(file2,  encoding='utf-8'))
columns = ['miRNA','sRNA_seq','count']
# columns = ['miRNA', 'target', 'score', 'mfe', 'mfe_ratio', 'start', 'gap', 'mismatch', 'GU', 'seed_gap', 'seed_mismatch', 'seed_GU', "miRNA_3'", 'aln', "target_5'", 'miRNA_bugle', 'target_bugle', 'tenth_aln', 'continuous_2_mismatch']
# ls = list(data2.columns)
# print(ls)
# a.rename(columns={'A':'a', 'C':'c'}, inplace = True)
data1.columns = columns
data = pd.merge(data1,data2, how='inner', on = ['miRNA'])
# data.to_csv('stu_pisRNA_all_targets_filtered_exp.csv', index=None)

col = ['miRNA', 'target', 'count', 'mfe','score', 'mfe_ratio',
       'start', 'gap', 'mismatch', 'GU', 'seed_gap',
       'seed_mismatch', 'seed_GU', 'sRNA_seq', "miRNA_3'", 'aln',
       "target_5'", 'miRNA_bugle', 'target_bugle', 'tenth_aln',
       'continuous_2_mismatch']
dat = data[col]
# dat.to_csv('stu_pisRNA_all_targets_filtered_exp.csv', index=None)
dat = dat.sort_values(axis = 0,ascending =[False,True,True] ,by = ['count', 'mfe','score'])
dat.to_csv('stu_pisRNA_all_targets_filtered_exp_sort.csv', index=None)