import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
# label-encoder转化器: 离散特征全部label-encoding （适合FM类）
def labelencode_trans(raw_df,dis_col,con_col,target):
    trans = ColumnTransformer(transformers=
                        [("label-encoder", OrdinalEncoder(), dis_col),   #  离散特征 labelEncoding. LabelEncoder()只支持一列，一般编码Y
                         ('con_col', 'passthrough', con_col),            #  连续特征
                         ('target','passthrough',[target])
                        ])
    values=trans.fit_transform(raw_df)                                   # fit

    # 输出转df。列名按照transformers处理的列名顺序组合
    dis_feature_names=trans.named_transformers_['label-encoder'].feature_names_in_  # 同输入
    col_feature_names=con_col                                                       # remiander列，按输入顺序输出
    columns=np.concatenate((dis_feature_names,col_feature_names,[target]))          # 最终列名。是transformer间顺序输出
    df=pd.DataFrame(values,columns=columns)                                         # 转成df

    #新离散/连续特征:
    new_dis_col=dis_feature_names
    new_con_col=col_feature_names

    # 转回原始列:每部分df分别转回原始列，再拼接 (只恢复特征部分，不管target)
    label_encoder=trans.named_transformers_['label-encoder']
    value1=label_encoder.inverse_transform(df[dis_feature_names])     # nparray
    input_features_1=label_encoder.feature_names_in_           # 原始输入特征

    values=np.concatenate((value1,df[col_feature_names].values),axis=1)
    orig_feature_names=np.concatenate((input_features_1,col_feature_names))

    raw_df2=pd.DataFrame(values,columns=orig_feature_names)

    cate_counts = dict(df[new_dis_col].nunique())  # 统计每个离散特征的不同取值个数
    # 统计每个离散特征取值被自动映射到的id: [0-C-1]
    cate_feature_map = dict()  # 统计离散特征每个原始值被映射到的id. 每个特征{取值：idx} 要是train/test都映射过了，就不需要这个字典了。直接index embedding就可以了
    feature_cates = label_encoder.categories_  # 每个域对应的feild取值.每个Field一个array
    for i, cate_name in enumerate(new_dis_col):  # 所有filed的原始域名称
        cate_feature_map[cate_name] = dict(zip(
            feature_cates[i], list(range(0, cate_counts[cate_name]))
        ))

    return trans,new_con_col,new_dis_col,df,raw_df2,cate_counts,cate_feature_map

# 只需要保存特征处理transformer和最终的dis_col,con_col(如果没有target)。 用来做转化，以及识别转换后的两类特征。
def test(test_df,trans,con_col,dis_col,target):
    test_values=trans.transform(test_df)

    # 转df (如果不需要，可以省略)
    dis_feature_names=trans.named_transformers_['label-encoder'].feature_names_in_  # 同输入
    col_feature_names=con_col                                                       # remiander列，按输入顺序输出
    columns=np.concatenate((dis_feature_names,col_feature_names,[target]))          # 最终列名。是transformer间顺序输出
    test_df=pd.DataFrame(test_values,columns=columns)
    return test_df