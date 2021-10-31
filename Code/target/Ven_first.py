import pandas as pd
import numpy as np
from matplotlib_venn import venn3
#超市
data_dwj_mf=pd.read_csv('Os_MF.csv')
#商友
data_sly_mf=pd.read_csv('sly_MFunction.csv')
data_dwj_kegg = pd.read_csv("Os_KEGG.csv")
data_sly_kegg = pd.read_csv("sly_KEGG.csv")
data_st_kegg = pd.read_csv("Potato_KEGG.csv")
# data_potato = pd.read_csv("../AAAAAAA_vens/potato_mapping_seq_len_count.csv")
# 百货
# data_baihuo=data_shangyou.loc[(data_shangyou['课室']!='超级市场课')|(data_shangyou['课室']!='生活美食课')]
#餐饮
# data_canyin=data_shangyou.loc[(data_shangyou['课室']=='生活美食课')|(data_shangyou['部类']=='B2F小型餐厅')|
#                               (data_shangyou['部类']=='小型餐厅')|(data_shangyou['部类']=='美食档口')|
#                               (data_shangyou['部类']=='甜品烘焙')|(data_shangyou['部类']=='休闲食品')]
#非可视化计算交叉情况
# print('超市',len(set(data_chaoshi['会员卡号'])))
# print('百货',len(set(data_baihuo['会员卡号'])))
# print('餐饮',len(set(data_canyin['会员卡号'])))
# print('超市&百货',len(set(data_chaoshi['会员卡号'])&set(data_baihuo['会员卡号'])))
# print('超市&餐饮',len(set(data_chaoshi['会员卡号'])&set(data_canyin['会员卡号'])))
# print('百货&餐饮',len(set(data_canyin['会员卡号'])&set(data_baihuo['会员卡号'])))
# print('超市&百货&餐饮',len(set(data_chaoshi['会员卡号'])&set(data_baihuo['会员卡号'])&set(data_canyin['会员卡号'])))
# print('仅超市',len(set(data_chaoshi['会员卡号'])-set(data_baihuo['会员卡号'])-set(data_canyin['会员卡号'])))
# print('仅百货',len(set(data_baihuo['会员卡号'])-set(data_chaoshi['会员卡号'])-set(data_canyin['会员卡号'])))
# print('仅餐饮',len(set(data_canyin['会员卡号'])-set(data_baihuo['会员卡号'])-set(data_chaoshi['会员卡号'])))

#开始绘制文氏图
import matplotlib.pyplot as plt
# 设置中文显示
plt.rcParams['font.sans-serif']='SimHei'
plt.rcParams['axes.unicode_minus']=False



# 导入库，注意没有安装的要先pip install matplotlib-venn

# sub接受一个set组成的列表,set_labels接受名称列表，其他参数自行去查看啦
# venn3(subsets=[set(data_chaoshi['会员卡号']),set(data_baihuo['会员卡号']),set(data_canyin['会员卡号'])],set_labels=['超市','百货','餐饮'],)
# plt.show()

# 如果是画两个集合的韦恩图，就以下代码,其他不变
fig, axs = plt.subplots(2,2, figsize=(6,6),dpi=300)
from matplotlib_venn import venn2, venn2_circles
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 50,
         }
g1=venn2(subsets=[set(data_dwj_mf['Pathway']),set(data_sly_mf['Pathway'])],
        set_labels=("Mo_MF","Bc_MF"),
        set_colors=("yellow","Purple"),
        alpha=0.4,#透明度
        normalize_to=1.0,#venn图占据figure的比例，1.0为占满
        ax=axs[0,0],
        )

# g1.get_label_by_id('01').set_fontsize(30)

# axs[0].annotate('81200',
#              color='#098154',
#              xy=g1.get_label_by_id('10').get_position(),
#              xytext=(-120, 100),
#              ha='center', textcoords='offset points',
#              bbox=dict(boxstyle='round,pad=0.5', fc='#098154', alpha=0.6),  # 注释文字底纹
#              arrowprops=dict(arrowstyle='-|>', connectionstyle='arc3,rad=0.5', color='#098154')  # 箭头属性设置
#              )

# chi1=axs[0].annotate('4842',
#              color='#c72e29',
#              xy=g1.get_label_by_id('01').get_position() + np.array([0, 0.05]),
#              xytext=(100, 150),
#              ha='center', textcoords='offset points',
#              bbox=dict(boxstyle='round,pad=0.5', fc='#c72e29', alpha=0.6),
#              arrowprops=dict(arrowstyle='-|>', connectionstyle='arc3,rad=0.5', color='#c72e29')
#              )
#
# chi2=axs[0].annotate('6099',
#              color='black',
#              xy=g1.get_label_by_id('11').get_position() + np.array([0, 0.05]),
#              xytext=(-50, 150),
#              ha='center', textcoords='offset points',
#              bbox=dict(boxstyle='round,pad=0.5', fc='grey', alpha=0.6),
#              arrowprops=dict(arrowstyle='-|>', connectionstyle='arc3,rad=-0.5', color='black')
#              )


g2=venn2(subsets=[set(data_dwj_kegg['Pathway']),set(data_sly_kegg['Pathway'])],
      set_labels=("Mo_KEGG","Bc_KEGG"),
      set_colors=("red","green"),
      alpha=0.4,
      normalize_to=1.0,
      ax=axs[0,1]
      )
# axs[1].annotate('I like this green part!',
#              color='#098154',
#              xy=g2.get_label_by_id('10').get_position() - np.array([0, 0.05]),
#              xytext=(-80, 40),
#              ha='center', textcoords='offset points',
#              bbox=dict(boxstyle='round,pad=0.5', fc='#098154', alpha=0.6),  # 注释文字底纹
#              arrowprops=dict(arrowstyle='-|>', connectionstyle='arc3,rad=0.5', color='#098154')  # 箭头属性设置
#              )

# chi3=axs[1].annotate('8076',
#              color='#c72e29',
#              xy=g2.get_label_by_id('01').get_position() + np.array([0, 0.05]),
#              xytext=(100, 150),
#              ha='center', textcoords='offset points',
#              bbox=dict(boxstyle='round,pad=0.5', fc='#c72e29', alpha=0.6),
#              arrowprops=dict(arrowstyle='-|>', connectionstyle='arc3,rad=0.5', color='#c72e29')
#              )
#
# chi4=axs[1].annotate('8477',
#              color='black',
#              xy=g2.get_label_by_id('11').get_position() + np.array([0, 0.05]),
#              xytext=(-80, 150),
#              weight= 50,
#              ha='center', textcoords='offset points',
#              bbox=dict(boxstyle='round,pad=0.5', fc='grey', alpha=0.6),
#              arrowprops=dict(arrowstyle='-|>', connectionstyle='arc3,rad=-0.5', color='black')
#              )
g3=venn2(subsets=[set(data_dwj_kegg['Pathway']),set(data_st_kegg['Pathway'])],
set_labels=("Mo_KEGG","Pi_KEGG"),
      set_colors=("red","blue"),
      alpha=0.4,
      normalize_to=1.0,
      ax=axs[1,0]
      )


g4=venn2(subsets=[set(data_sly_kegg['Pathway']),set(data_st_kegg['Pathway'])],
set_labels=("Bc_KEGG","Pi_KEGG"),
      set_colors=("green","blue"),
      alpha=0.4,
      normalize_to=1.0,
      ax=axs[1,1]
      )

# axs[2].annotate('I like this green part!',
#              color='#098154',
#              xy=g3.get_label_by_id('10').get_position() - np.array([0, 0.05]),
#              xytext=(-80, 40),
#              ha='center', textcoords='offset points',
#              bbox=dict(boxstyle='round,pad=0.5', fc='#098154', alpha=0.6),  # 注释文字底纹
#              arrowprops=dict(arrowstyle='-|>', connectionstyle='arc3,rad=0.5', color='#098154')  # 箭头属性设置
#              )

# chi5=axs[2].annotate('63831',
#              color='#c72e29',
#              xy=g3.get_label_by_id('01').get_position() + np.array([0, 0.05]),
#              xytext=(80, 150),
#              weight = 'black',
#              ha='center', textcoords='offset points',
#              bbox=dict(boxstyle='round,pad=0.5', fc='#c72e29', alpha=0.6),
#              arrowprops=dict(arrowstyle='-|>', connectionstyle='arc3,rad=0.5', color='#c72e29')
#              )
#
# chi6=axs[2].annotate('23237',
#              color='black',
#              xy=g3.get_label_by_id('11').get_position() + np.array([0, 0.05]),
#              xytext=(-60, 150),
#              ha='center', textcoords='offset points',
#              bbox=dict(boxstyle='round,pad=0.5', fc='grey', alpha=0.6),
#              arrowprops=dict(arrowstyle='-|>', connectionstyle='arc3,rad=-0.5', color='black')
#              )
# chi1.set_fontsize(40)
# chi2.set_fontsize(40)
# chi3.set_fontsize(40)
# chi4.set_fontsize(40)
# chi5.set_fontsize(40)
# chi6.set_fontsize(40)
# plt.subplots_adjust(left=0.1, bottom=0.5, right=0.8, wspace=0.01)
# legend = plt.figlegend(prop=font1)
# axs.set_labels( fontsize =15)
g1.get_label_by_id('10').set_fontsize(15)#1的大小设置为20
g1.get_label_by_id('01').set_fontsize(15)#1的大小设置为20
g1.get_label_by_id('11').set_fontsize(15)#1的大小设置为20
g2.get_label_by_id('10').set_fontsize(15)#1的大小设置为20
g2.get_label_by_id('01').set_fontsize(15)#1的大小设置为20
g2.get_label_by_id('11').set_fontsize(15)#1的大小设置为20
g3.get_label_by_id('10').set_fontsize(15)#1的大小设置为20
g3.get_label_by_id('01').set_fontsize(15)#1的大小设置为20
g3.get_label_by_id('11').set_fontsize(15)#1的大小设置为20
g4.get_label_by_id('10').set_fontsize(15)
g4.get_label_by_id('01').set_fontsize(15)
g4.get_label_by_id('11').set_fontsize(15)


for text in g1.set_labels:
    text.set_fontsize(15)
for text in g2.set_labels:
    text.set_fontsize(15)
for text in g3.set_labels:
    text.set_fontsize(15)
for text in g4.set_labels:
    text.set_fontsize(15)

# plt.tight_layout(pad=0.5)

plt.savefig(('713.png'), dpi=300)
plt.show()


