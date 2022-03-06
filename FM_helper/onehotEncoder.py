# onehot-encode转化器。
def onehot_trans(raw_df,dis_col,con_col,target):
    # 取值小于10的离散特征one-hot     (符合lr,xgb等模型。但数值特征最好分箱后one-hot. 高基特征最好target-coding)
    dis_col_less_10 = raw_df[dis_col].nunique()[raw_df[dis_col].nunique() <= 10].index  # 这部分可以考虑one-hot
    dis_col_more_10 = [i for i in dis_col if i not in dis_col_less_10]
    trans =  ColumnTransformer(transformers=
                        [("one-hot", OneHotEncoder(handle_unknown='ignore',sparse=False), dis_col_less_10),# one-hot  
                         ("label-encoder", OrdinalEncoder(), dis_col_more_10), # 剩余离散特征
                         ('con_col', 'passthrough', con_col),                 #  连续特征
                         ('target','passthrough',[target])
                        ])
                        #remainder='passthrough')        # 剩下的全都顺序输出 
    values=trans.fit_transform(raw_df)                  # ColumnTransformer按每个transformer顺序输出。且是array

    # 输出转df。 列名按照transformers处理的列名顺序组合
    onehot_feature_names=trans.named_transformers_['one-hot'].get_feature_names_out(dis_col_less_10) # one-hot后的列名
    dis_feature_names=trans.named_transformers_['label-encoder'].feature_names_in_  # label-encode列。列名不变
    col_feature_names=con_col

    columns=np.concatenate((onehot_feature_names,dis_feature_names,col_feature_names,[target])) # 最终列名
    df=pd.DataFrame(values,columns=columns)

    #新离散/连续特征:
    new_dis_col=dis_feature_names
    new_con_col=np.concatenate((onehot_feature_names,col_feature_names)) # one-hot的之后当连续特征看

    # 转回原始列:每部分df分别转回原始列，再拼接 (只恢复特征部分，不管target)
    onehot_encoder=trans.named_transformers_['one-hot']
    value1=onehot_encoder.inverse_transform(df[onehot_feature_names]) # nparray
    input_features_1=onehot_encoder.feature_names_in_                 # 原始输入特征1
    label_encoder=trans.named_transformers_['label-encoder']
    value2=label_encoder.inverse_transform(df[dis_feature_names])     # nparray
    input_features_2=label_encoder.feature_names_in_                  # 原始输入特征2
    values=np.concatenate((value1,value2,df[col_feature_names].values),axis=1)   
    orig_feature_names=np.concatenate((input_features_1,input_features_2,col_feature_names))
    raw_df2=pd.DataFrame(values,columns=orig_feature_names)
    return new_con_col,new_dis_col,df,raw_df2
