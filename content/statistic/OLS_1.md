---
title: "[Just For Fun 😄]收入影响对社会的信任？ Simple Linear Regression"
date: 2023-06-19T00:06:11+09:00
description: 用单变量线性回归对KGSS 2021年的调查数据中 收入 与 社会信任 两个变量间的线性关系进行分析
draft: false
---

# [Just For Fun 😄]收入影响对社会的信任？ Simple Linear Regression

{{<hint warning>}}
【Just For Fun 😄】: 所有标有此标签的文章均**无法保证**文章内容质量的科研严谨性， 仅作为技术分享
{{</hint>}}

本文对KGSS 2021年的调查数据中的 `RINCOM0`收入变量 和`RELIABLE`社会信任变量 间的线性关系进行统计检验

## Method

- 假设 Y：`RELIABLE` X：`RINCOM0`
- 使用Python包 statsmodels 中的 OLS 线性回归函数

## Python Code


```python
import pandas as pd
import statsmodels.api as sm
import numpy as np
```

### Load Data from SPSS FILE

*Pandas现在支持直接通过 read_SPSS导入SPSS文件了 喜大普奔* 🫡

```python
df = pd.read_spss("2003-2021_KGSS_public_v3_20230210.sav", convert_categoricals=False)  #由于文件太大 直接不导入SPSS的label label通过KGSS官网给出的Codebook就可以确认了

```

由于KGSS给出的SPSS数据是多年份统合数据，因此要先提取出2021年的数据

```python
df_2021 = df[df["YEAR"]==2021]
```

接下来就是日常处理流程

1. 根据Codebook确定变量名
2. 通过 `value_counts()` 确定有无异常值
3. 替换异常值
4. 利用statemodels包的`OLS`函数 进行线性回归分析并检验
5. 分析检验统计量

### Pre-Process


先看看自变量 收入 `RINCOM0`
```python
RINCOM = df_2021.RINCOM0
RINCOM.describe() #其包含-1,-8的异常值
```

    count    1205.000000
    mean      174.807469
    std       211.070790
    min        -8.000000
    25%        -1.000000
    50%       150.000000
    75%       300.000000
    max      2400.000000
    Name: RINCOM0, dtype: float64

可以看出 其包含-8和-1的异常值 因为人的收入不可能是负数😂 因此直接替换为空(`np.nan`)就可以了

```python
RINCOM.replace(to_replace=[-1,-8], value=np.nan, inplace=True)
RINCOM.describe()
```

    count     730.000000
    mean      289.547945
    std       200.331463
    min         0.000000
    25%       190.000000
    50%       250.000000
    75%       350.000000
    max      2400.000000
    Name: RINCOM0, dtype: float64



对于因变量 `RELIABLE`则没有任何异常值

```python
df_2021.RELIABLE.value_counts() #查看回答分布，可以看出没有异常值(-8)[不知道]的回答
```
### OLS

```python
sm.OLS(RELIABLE, RINCOM, missing='drop').fit().summary()
```

## Result

    OLS Regression Results
    Dep. Variable:	RELIABLE	R-squared (uncentered):	0.651
    Model:	OLS	                Adj. R-squared (uncentered):	0.650
    Method:	Least Squares	    F-statistic:	1357.
    Date:	Sun, 18 Jun 2023	Prob (F-statistic):	1.29e-168
    Time:	23:57:42	        Log-Likelihood:	-1979.9
    No. Observations:	730	    AIC:	3962.
    Df Residuals:	729	        BIC:	3966.
    Df Model:	1		
    Covariance Type:	nonrobust		
            coef	std err	    t	P>|t|	[0.025	0.975]
    RINCOM0	0.0141	0.000	36.843	0.000	0.013	0.015
    Omnibus:	442.021	        Durbin-Watson:	1.256
    Prob(Omnibus):	0.000	    Jarque-Bera (JB):	8265.470
    Skew:	-2.351	            Prob(JB):	0.00
    Kurtosis:	18.800	        Cond. No.	1.00


    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.


首先$r^2$为 0.651 
因变量RINCOM0系数为 0.0141
t检验概率为0

可得**人收入越高💰 越信任这个社会🤝**


（ps. 果然，挣得多连世界都是美妙的👀）