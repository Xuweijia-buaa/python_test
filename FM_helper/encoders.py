onehot_trans =  ColumnTransformer(transformers=
                        [("one-hot", OneHotEncoder(handle_unknown='ignore',sparse=False), dis_col_less_10),# LabelBinarizer().只支持一列，一般对label进行one-hot:one-vs-others  
                         ("label-encoder", OrdinalEncoder(), dis_col_more_10)],  # LabelEncoder()只支持一列, 一般编码Y取C个
                        remainder='passthrough')
labelencode_trans = ColumnTransformer(transformers=
                        [("label-encoder", OrdinalEncoder(), dis_col)],          # LabelEncoder()只支持一列，一般编码Y取C个
                        remainder='passthrough')
# 如果想同一列加工出不同特征。可以用FeatureUnion和自定义transformer来选择列。 ColumnTransformer对同一列只能做一个操作。不对同一列做不同操作，就用后者。如加工不同类型特征（对text列同时加工文本长度和文本tfidf2特征）
