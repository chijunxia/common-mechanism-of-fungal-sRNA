import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
matplotlib.rcParams['font.family']='SimHei'

font = {'family': 'Times New Roman',
            'color': 'black',
            'weight': 'normal',
            'size': 20,
            }

def find_index(ls, fls):
    maplist = []
    for x in fls:
        yy = ls[0]
        dd = abs(x - yy)
        for y in ls:
            d = abs(x - y)
            if dd > d:
                yy, dd = y, d
        index = ls.index(yy)
        maplist.append(index)
    return maplist


def getcolarmap(vmin, vmax, vlist): #柱状图从下往上画
    dict = {}
    colormap = []
    colorlist = []
    sep = 1000
    colors = cm.jet(np.linspace(0, 1, sep))  # 初始应该是0-1区间 (自己做一个bar)
    # print(colors)
    d = (vmax - vmin)/sep  # 间隔一致
    x = vmin
    for i in colors:
        colormap.append(x)
        x = x + d
        # colormap.append(x)
    # print(colors)
    maplist = find_index(colormap, vlist)
    # print(maplist)
    for element in maplist:
        colorlist.append(colors[element]) #获取对应的颜色

    sm = matplotlib.cm.ScalarMappable(cmap=cm.get_cmap('jet'))
    sm.set_array([vmin, vmax])  # 调整刻度
    # print(colorlist)
    return colorlist, sm




###########################      输出处理       ####################################################

# file = 'bic_dup_level_6_bp.csv'
file = 'Potato_OS_sly_KEGG.csv'
# top = 50
# start = time.clock()
data = pd.DataFrame(pd.read_csv(file, encoding='utf-8' ))
# data = data.sort_values(by='cluster_score', ascending=False)
# data = data.head(top)
# data.to_csv(('GO analysis/'+file).replace('.csv','_top'+str(top)+'.csv'),index=None)


# fig =plt.figure(1)
# pathway ID,pathway description,observed gene count,false discovery rate,
fig, ax = plt.subplots(1,1, figsize=(15,9))
plt.ylabel("Observed gene count",font)
# plt.ylabel("Pathway")
#

name_list = list(data['term description']) #best_go  GO_desc
ratio_list1 = list(data['observed gene count_x'])
ratio_list2 = list(data['observed gene count_y'])
ratio_list3 = list(data['observed gene count_x1'])
p_list1 = list(data['false discovery rate_x'])
p_list2 = list(data['false discovery rate_y'])
p_list3 = list(data['false discovery rate'])

vmin = 0.00
vmax = 0.05

colors1,sm = getcolarmap(vmin, vmax, p_list1)
colors2,sm = getcolarmap(vmin, vmax, p_list2)
colors3,sm = getcolarmap(vmin, vmax, p_list3)
cb=fig.colorbar(sm, label="False discovery rate")
cb.set_label('False discovery rate',fontdict=font)
#
cb.ax.tick_params(labelsize=16)
# print(name_list)
# print(p_list)
bar_width=0.26
bar_width1= 0.52

x_index=np.arange(9)
# plt.bar(x_index, ratio_list1,tick_label = name_list, color= colors1, width= 0.3, edgecolor='black')
# plt.bar(x_index+bar_width, ratio_list2,tick_label = name_list, color= colors2, width= 0.3, edgecolor='black')

#迟君霞改
# max_chars = 10
#
# new_labels = ['\n'.join(name_list._text[i:i + max_chars ]
#                         for i in range(0, len(label._text), max_chars ))
#               for label in ax.get_yticklabels()]
#
# ax.set_yticklabels(new_labels)
# plt.barh(y_index , ratio_list3,tick_label = name_list, color= colors3, height= 0.25, edgecolor='black',align = 'edge',orientation="horizontal")
# plt.barh(y_index+ bar_width , ratio_list2,tick_label = name_list, color= colors2, height = 0.25, edgecolor='black',align = 'edge',orientation="horizontal")
# plt.barh(y_index + bar_width1, ratio_list1,tick_label = name_list, color= colors1,height=0.25,edgecolor='black',align = 'edge',orientation="horizontal")
plt.bar(x_index , ratio_list3,tick_label = name_list, color= colors3, width= 0.25, edgecolor='black')
plt.bar(x_index+ bar_width , ratio_list2,tick_label = name_list, color= colors2, width = 0.25, edgecolor='black')
plt.bar(x_index + bar_width1, ratio_list1,tick_label = name_list, color= colors1,width=0.25,edgecolor='black')
# ax.set_xticklabels(name_list, rotation=90,fontdict={'family': 'Times New Roman',
#             'color': 'black',
#             'weight': 'normal',
#             'size': 5,
#             })
ax.set_xticklabels(name_list, rotation=45,fontdict=font)
#
plt.tick_params(labelsize=20)
label_y = ax.get_yticklabels()
# max_chars = 33
#
# new_labels = ['\n'.join(name_list._text[i:i + max_chars ]
#                         for i in range(0, len(name_list._text), max_chars ))
#               for name_list in ax.get_yticklabels()]

# ax.set_yticklabels(new_labels)

# plt.setp(label_y, rotation=45, horizontalalignment='right', size=10)
#迟君霞
# plt.setp(label_y, horizontalalignment='right', size=20)
# ax.set_xlabel('Genes', fontdict=font)
    # label_x = ax.get_xticklabels()
# a = plt.barh(range(len(ratio_list)), ratio_list,tick_label = name_list,height= 0.5,color= colors ,edgecolor='blue')
# fig.colorbar(sm, label="false discovery rate")
fig.tight_layout(pad=3)   ###名字出边界 p边,w水平,h竖直
plt.savefig(('Os_sly_Go_20.png'), dpi=300)
plt.show()


end = time.clock()
# print('Running dict time: %s Seconds'%(end-start))


