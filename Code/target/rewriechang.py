import pandas as pd
from collections import Counter
def filter_data(file, outfile):
    data = pd.DataFrame(pd.read_csv(file))
    data.columns = ['seq', 'len']
    data = data[data['len'] >= 18]
    data = data[data['len'] <= 25]
    # print(data.size)
    print(len(data))
    data.to_csv(outfile, index=None)
    return True

def get_inner(df1, df2):

    #seqs1=df1['seq']
    #seqs2=df2['seq']
    #seqs1 = set(df1['seq'])
    #seqs2 = set(df2['seq'])
    #innerseqs = seqs1 & seqs2
    #print(len(innerseqs))
    #print(innerseqs)

    innerseqs=pd.merge(df1, df2, on=['#term ID'])#求交集
    print(len(innerseqs))
    print(innerseqs)
    return innerseqs
    ################



    #求补集即差集
# def get_inner(df1, df2):
#     df1 = df1.append(df2)
#     df1 = df1.append(df2)
#     df1 = df1.drop_duplicates(subset=['sRNA'], keep=False)
#     print(len(df1))
#     print(df1)
#     #print(len(innerseqs))
#     #print(innerseqs)
#     return df1
#     #return innerseqs
#
#     ######################
#     #求并集
'''

    # union=pd.merge(df1, df2, on=['seq'],how='outer')
    # print(len(union))
    # print(union)
    # return union


'''
# def sum_ls(ls):
#
#     :param ls: list
#     :return:  summary
#
#     x = Counter(ls)
#     sum_ls = x.most_common()
#     # ======================================================
#     # sum_ls = [('ACCCCC', 2), ('B', 4)]
#     # for record in sum_ls:
#     #     ls = [record[0], record[1], len(record[0])]inner
#     #     print(ls)
#     # ======================================================
#     return sum_ls
# '''
if __name__ == '__main__':
    # step 1
    # get_count('GSM1101909_small_RNA_Bc_total.txt', 'small_RNA_Bc_total.csv')
    # get_count('GSM1101915_small_RNA_sly_leaf_bc_72hpi.txt', 'small_RNA_sly_total.csv')

    # step 2
    # filter_data('small_RNA_sly_total.csv', 'small_RNA_sly_filtered.csv')      #  812188
    # filter_data('small_RNA_Bc_total.csv', 'small_RNA_Bc_filtered.csv')        # 1534986

    # step 3
    df1 = pd.DataFrame(pd.read_csv('MF.csv'))
    df2 = pd.DataFrame(pd.read_csv('sly_MFunction.csv'))
    out = "Os_sly_MF1.csv"
    innerseqs = get_inner(df1, df2)
    innerseqs.to_csv(out, index=None)

    # union = pd.merge(df1, df2, on=['sequence'], how='outer')
    # print(len(union))
    # union.to_csv(out,index=None)


