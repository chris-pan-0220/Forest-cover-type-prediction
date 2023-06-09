# Forest-cover-type-prediction
https://www.kaggle.com/competitions/forest-cover-type-prediction/overview/evaluation

# referece

# EDA 

### seaborn

[color palettes](https://seaborn.pydata.org/tutorial/color_palettes.html)

# feature engineering
[8 Feature Engineering Techniques for Machine Learning](https://www.projectpro.io/article/8-feature-engineering-techniques-for-machine-learning/423)

[7 of the Most Used Feature Engineering Techniques](https://towardsdatascience.com/7-of-the-most-used-feature-engineering-techniques-bcc50f48474d)


[特徵工程到底是什麼？](https://medium.com/ai%E5%8F%8D%E6%96%97%E5%9F%8E/%E7%89%B9%E5%BE%B5%E5%B7%A5%E7%A8%8B%E5%88%B0%E5%BA%95%E6%98%AF%E4%BB%80%E9%BA%BC-%E8%BD%89%E9%8C%84-ca9b82b7b646)

### transformation 

[What could be the reason for using square root transformation on data?](https://stats.stackexchange.com/questions/11359/what-could-be-the-reason-for-using-square-root-transformation-on-data)

### bining

* binning
    * [数据科学猫：数据预处理 之 数据分箱(Binning)](https://blog.csdn.net/Orange_Spotty_Cat/article/details/116485079)
    * [数据分箱方法 woe编码_功能工程深入研究编码和分箱技术](https://blog.csdn.net/weixin_26704853/article/details/108892251)

### clustering
* scipy clustering

https://medium.com/ai-academy-taiwan/clustering-method-4-ed927a5b4377

[scipy.cluster.hierarchy 学习 & 总结 (fcluster, linkage等)](https://blog.csdn.net/Petersburg/article/details/121981388)


[scipy.cluster.hierarchy.linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)

[用Python做层次聚类分析](https://zhuanlan.zhihu.com/p/37374120)

[scipy中squareform函数详解](https://blog.csdn.net/counsellor/article/details/79555619)

### missing value 
[【Python数据分析基础】: 数据缺失值处理](https://zhuanlan.zhihu.com/p/40775756)
[【Python機器學習】109：當數據集含有遺漏值的處理方法與未經過訓練的分類預測器](https://medium.com/%E5%B1%95%E9%96%8B%E6%95%B8%E6%93%9A%E4%BA%BA%E7%94%9F/python%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-109-%E7%95%B6%E6%95%B8%E6%93%9A%E9%9B%86%E5%90%AB%E6%9C%89%E9%81%BA%E6%BC%8F%E5%80%BC%E7%9A%84%E8%99%95%E7%90%86%E6%96%B9%E6%B3%95%E8%88%87%E6%9C%AA%E7%B6%93%E9%81%8E%E8%A8%93%E7%B7%B4%E7%9A%84%E5%88%86%E9%A1%9E%E9%A0%90%E6%B8%AC%E5%99%A8-addda5ee72a0)

### feature encoding 
[不要再做One Hot Encoding！！](https://axk51013.medium.com/%E4%B8%8D%E8%A6%81%E5%86%8D%E5%81%9Aone-hot-encoding-b5126d3f8a63)

[Encoding:机器学习中类别变量的编码方法总结](https://zhuanlan.zhihu.com/p/514937526)

[Kaggle Categorical Encoding 3大絕招](https://axk51013.medium.com/kaggle-categorical-encoding-3%E5%A4%A7%E7%B5%95%E6%8B%9B-589780119470)

[Categorical Encoding](https://ithelp.ithome.com.tw/articles/10272126)

[sklearn中多种编码方式——category_encoders](https://blog.csdn.net/sinat_26917383/article/details/107851162)

### feature selection 

[scikit-learn中的特征选择方法](https://zhuanlan.zhihu.com/p/141506312)

[【机器学习】特征选择(Feature Selection)方法汇总](https://zhuanlan.zhihu.com/p/74198735)


* filter method 
    * [Sklearn 卡方检验](https://zhuanlan.zhihu.com/p/357801038)
    * [sklearn.feature_selection chi2基于卡方，特征筛选详解](https://blog.csdn.net/u013066730/article/details/110952738)

* wrapper method
    * [Feature Selection -- 2. Wrapper Methods(包裝器法)](https://ithelp.ithome.com.tw/articles/10246251)

* Genetic search 
    * [特征搜索](https://leoncuhk.gitbooks.io/feature-engineering/content/feature-building01.html)
    * [基因演算法的世界](https://ithelp.ithome.com.tw/users/20111679/ironman/2577)
    * [Feature Selection with Genetic Algorithms](https://towardsdatascience.com/feature-selection-with-genetic-algorithms-7dd7e02dd237)
    * [遗传算法入门到掌握](https://blog.csdn.net/qq_34374664/article/details/78874956)
    * [遗传算法调参 参数设置](https://blog.csdn.net/carlyll/article/details/105900317)


[hyperopt for feature selection](https://celmore25.medium.com/automated-feature-selection-with-hyperopt-46865e1b4fce)


### feature (dimension) reduction
[機器學習: 降維(Dimension Reduction)- 線性區別分析( Linear Discriminant Analysis)](https://chih-sheng-huang821.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E9%99%8D%E7%B6%AD-dimension-reduction-%E7%B7%9A%E6%80%A7%E5%8D%80%E5%88%A5%E5%88%86%E6%9E%90-linear-discriminant-analysis-d4c40c4cf937)


# training 

### train test split 

[训练集(train set) 验证集(validation set) 测试集(test set)](https://blog.csdn.net/zwqjoy/article/details/78788924)

### xgboost 

[xgboost 之 eval_metric参数的应用以及构造损失函数的变化情况图](https://blog.csdn.net/qq_35307209/article/details/89914785)

[How Does XGBoost Handle Multiclass Classification?](https://towardsdatascience.com/how-does-xgboost-handle-multiclass-classification-6c76ba71f6f0)

[机器学习系列(12)_XGBoost参数调优完全指南（附Python代码）](https://blog.csdn.net/han_xiaoyang/article/details/52665396)

[How to use early stopping in Xgboost training?](https://mljar.com/blog/xgboost-early-stopping/)

### lightgbm

[LightGBM+OPTUNA超参数自动调优教程](https://zhuanlan.zhihu.com/p/409535386)

[GBM Hyperparameter Tuning with Optuna](https://www.kaggle.com/code/saurabhshahane/lgbm-hyperparameter-tuning-with-optuna-beginners)

### kaggle reference 
[PSS3E10 #7 Winning Model](https://www.kaggle.com/code/ambrosm/pss3e10-7-winning-model)

[Forest Cover Type - Feature Engineering](https://www.kaggle.com/code/rsizem2/forest-cover-type-feature-engineering/notebook)


[Forest_Prediction_Final](https://www.kaggle.com/code/nehabhandari1/forest-prediction-final)

# validation 

[机器学习最优模型选择详解](https://zhuanlan.zhihu.com/p/404151451)
[[Day29]機器學習：交叉驗證！](https://ithelp.ithome.com.tw/articles/10197461)
[sklearn 学习曲线Learning Curve和 validation_curve](https://blog.csdn.net/vincent_duan/article/details/121270138)

# tuning 
[[机器学习] 超参数优化介绍](https://zengwenqi.blog.csdn.net/article/details/89373581?ydreferer=aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3p3cWpveS9hcnRpY2xlL2RldGFpbHMvNzg3ODg5MjQ%3D)

[bayesian-optimization-with-optuna-stacking](https://www.kaggle.com/code/dixhom/bayesian-optimization-with-optuna-stacking)

[A Guide to Find the Best Boosting Model using Bayesian Hyperparameter Tuning but without Overfitting](https://towardsdatascience.com/a-guide-to-find-the-best-boosting-model-using-bayesian-hyperparameter-tuning-but-without-c98b6a1ecac8)

[調整模型超參數利器 - Optuna](https://ithelp.ithome.com.tw/articles/10276835)

# ensembling 

### stacking 

[集成学习 (Ensemble Learning) (四) —— Stacking 与 Blending](https://blog.csdn.net/qq_39478403/article/details/112775097)

[Pulsar | Ensemble of XGB,LGBM,CATB,XGBR | Fraction](https://www.kaggle.com/code/sahilsg/pulsar-ensemble-of-xgb-lgbm-catb-xgbr-fraction)

[Stacking：集成学习策略图解](https://blog.csdn.net/qq_35509823/article/details/103833482)