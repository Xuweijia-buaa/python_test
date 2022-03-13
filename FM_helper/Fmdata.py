# 自定义transformer.假设前边已经做好了缺失值补全(包括“”)，特征选择，数值型特征归一化 （也是用transformer）.
# 需要前边的缺失值是真的缺失值（不做贡献，而非Y/N那种）。这里所有NAN都不会被用到，作为0处理。
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import os


class FeaturePosTrans(BaseEstimator, TransformerMixin):
    def __init__(self, dis_col=None, con_col=None, limit_freq=0):
        self.dis_col = dis_col
        self.con_col = con_col
        self.limit_freq = limit_freq

        self.NULL = '<NULL>'
        self.UNK = '<UNK>'  # nlp里。低频是1，NAN是0. NAN作为padding,不参与训练且是0
        # NAN对应embedding： padding_index=0.只占位，不训练）
        # nn.Embedding(V,d,padding_idx=0

        self.dis_col_map = dict()  # 按特征，记录取值到位置id的映射  只用来存着
        self.feature_id_map = dict()  # 特征名到位置id的映射大表 {特征名_取值：位置id}
        self.pos = 0  # 位置id
        self.dis_col_count = dict()  # 每个离散特征的取值数目

        # 所有离散的缺失值，统一用NAN编码，之后在w，E中padding成0
        self.feature_id_map[self.NULL] = 0
        self.pos += 1

        if (con_col != None):
            self.feature_id_map.update(dict(zip(con_col, range(self.pos, self.pos + len(con_col)))))  # 连续特征到对应位置的映射
            self.pos += len(self.con_col)

    def fit(self, X, y=None):

        if (self.dis_col != None):
            # 每个离散特征取值,映射到对应id
            for col in self.dis_col:
                valueCount = dict(X[col].value_counts())  # 该离散特征。每个取值的出现数目
                # 是否特殊处理低频取值
                if self.limit_freq > 0:
                    values = [k for k, v in valueCount.items() if k != self.NULL and v > self.limit_freq]  # 该特征留下的取值
                    self.dis_col_map[col] = dict(zip(values, range(self.pos + 1, self.pos + 1 + len(values))))
                    self.dis_col_map[col][self.UNK] = self.pos
                    # 组织大表。类似
                    new_values = [col + "_" + v for v in values]  # { C1_v1：id}
                    self.feature_id_map.update(
                        dict(zip(new_values, range(self.pos + 1, self.pos + 1 + len(new_values)))))
                    self.feature_id_map[col + "_" + self.UNK] = self.pos
                    self.pos += len(new_values) + 1  # 每个特征留下：所有高频取值+UNK
                else:
                    # 每个特征。分别记录映射
                    values = [k for k in valueCount.keys() if k != self.NULL]  # 该离散特征所有取值（除缺失值）
                    self.dis_col_map[col] = dict(zip(values, range(self.pos, self.pos + len(values))))
                    # 类似，但根据取值记在大map里
                    new_values = [col + "_" + v for v in values]  # { C1_v1：id}
                    self.feature_id_map.update(dict(zip(new_values, range(self.pos, self.pos + len(new_values)))))
                    self.pos += len(new_values)

                # 每个离散特征的有效取值数目(不含NAN，含每个特征的unk)
                self.dis_col_count[col] = len(self.dis_col_map[col])

    def transform(self, X, label=None):
        # 映射：
        feature_pos = X.copy()  # 样本每个特征对应的位置
        feature_values = X.copy()  # 样本每个特征的取值。离散特征取值是1.
        cols = self.dis_col + self.con_col

        # 如果有target列，删掉target列
        if label in feature_pos.columns:
            feature_pos = feature_pos.drop([label], axis=1)
            feature_values = feature_values.drop([label], axis=1)  # 特征列去掉label

        for col in cols:
            if col in self.dis_col:
                # values=X[col].apply(self.gen,args=(col,)).values
                values = X[col].apply(self.gen2, args=(col,)).values  # 组织形式不同。映射效果相同。用这个好些
                feature_pos[col] = values
                feature_values[col] = 1.0
            else:
                feature_pos[col] = self.feature_id_map[col]  # 连续特征取值不变  。 位置是映射后的id

        # 映射完的取值（包括离散特征取值1.0），也都变成float32
        feature_values = feature_values.astype(np.float32)

        return feature_pos, feature_values

    # 如果是多列。传入的x是该列对应的series. 输出的是这些列拼起来的df
    # 如果是单列，传入的x是该列的每个元素    输出的是该列对应的Series
    # 根据离散特征取值，返回对应的位置id
    def gen(self, x, col):
        if x == self.NULL:  # NAN统一映射到0
            return 0
        else:
            if x in self.dis_col_map[col]:
                return self.dis_col_map[col][x]  # 按取值，映射到对应位置id
            else:
                if self.limit_freq > 0:
                    return self.dis_col_map[col][self.UNK]  # 低频取值/没见过的值。映射到unqkey对应的编码
                else:
                    return 0  # 没见过的值。映射到NAN==0。没有贡献

    # 用大表做映射。类似
    def gen2(self, x, col):
        if x == self.NULL:  # NAN统一映射到0
            return 0
        else:
            x = col + "_" + x
            if x in self.feature_id_map:  # 其他按取值，映射到对应位置id
                return self.feature_id_map[x]
            else:
                if self.limit_freq > 0:
                    return self.feature_id_map[col + "_" + self.UNK]  # 低频取值/没见过的值。映射到该特征unqkey对应的编码
                else:
                    return 0  # 没见过的值。映射到NAN。没有贡献

    def id2name(self):
        return dict(zip(self.feature_id_map.values(), self.feature_id_map.keys()))


# 分别找出连续列/离散列
def col_type(df):
    dis_col=[]
    con_col=[]
    columns=df.columns.tolist()
    for c in columns:
        if df[c].dtype=='int64' or df[c].dtype=='float':
            con_col.append(c)
        else:
            dis_col.append(c)
    return dis_col,con_col

if __name__ == '__main__':
    data_path = '/media/xuweijia/DATA/代码/python_test/data/Criteo/demo_data/'
    file_name = 'train.csv'
    raw_df = pd.read_csv(os.path.join(data_path + file_name))
    raw_df = raw_df.drop(["Id"], axis=1)

    dis_col, con_col = col_type(raw_df)
    con_col.remove("Label")
    target = "Label"

    null_token = '<NULL>'
    raw_df[dis_col] = raw_df[dis_col].fillna(null_token)
    raw_df[con_col] = raw_df[con_col].fillna(0)

    f_trans=FeaturePosTrans(dis_col,con_col,10)  # 保存映射字典。按原始特征统计的每个特征取值。 以及对应数目。
    f_trans.fit(raw_df)                          # NAN统一映射到位置0： w0, embedding0. 不参与训练。取值0.不起作用
    # {'<NULL>': 0,                              # 每个特征的低频取值(训练集中出现次数<某阈值)，被映射成特征里的UNK。对应一个embedding，统一训练
    #  'I1': 1,
    #  'I2': 2,
    #  'I3': 3,
    #  'I4': 4,
    #  'I5': 5,
    #  'I6': 6,
    #  'I7': 7,
    #  'I8': 8,
    #  'I9': 9,
    #  'I10': 10,
    #  'I11': 11,
    #  'I12': 12,
    #  'I13': 13,
    #  'C1_<UNK>': 14
    #  'C1_05db9164': 15,
    #  'C1_68fd1e64': 16
    #  ...
    #  }

    test_df = raw_df.copy()
    test_df["C22"] = ["1"] * 1500 + ['<NULL>'] * 99  # 没见过的值和空值

    # 根据映射字典，返回样本每个特征的位置id(连续特征根据名称映射，离散特征根据名称+取值映射).  （m,n） 原始特征数目
    # 和每个特征取值（离散是1，连续不变）（m,n）原始特征数目
    feature_pos, feature_values = f_trans.transform(test_df)

