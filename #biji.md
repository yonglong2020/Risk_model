# 机器学习软件环境配置
## 一、下载并安装 Anaconda


## 二、创建并激活专属虚拟环境
```bash
conda create -n risk_model python=3.11
conda activate risk_model

#补充
conda --version #查看conda版本
conda env list #查看当前有哪些环境
where python #验证当前使用的 Python 位置
conda deactivate  #退出虚拟环境 / 回到默认环境
conda remove -n risk_model --all #彻底删除某个环境
```


## 三、安装核心库
```bash
pip install xgboost -i https://pypi.tuna.tsinghua.edu.cn/simple
#注：如果使用 conda install，通常会顺便帮你安装好 GPU 依赖（如 cudatoolkit），但如果只想用 CPU 训练，pip 版本足够且稳定。

pip install numpy pandas scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
#numpy：Python 的数值计算基础库。提供了“数组”这种高效的数据结构，所有科学计算都建立在它之上。
#pandas：数据分析的核心库。它提供“表格”（DataFrame），你可以像用 Excel 一样筛选、合并、分组数据。机器学习第一步就是把原始数据变成 Pandas 表格。
#scikit-learn：机器学习“工具箱”。包含数据预处理（标准化、编码分类变量）、模型评估（交叉验证、网格搜索）、常用算法（线性回归、随机森林等）。它和 XGBoost 配合得很好，用来做特征工程和模型验证。scikit-learn 不仅用于数据预处理（如 StandardScaler、LabelEncoder），还用于模型评估（交叉验证、网格搜索）。

pip install matplotlib seaborn graphviz -i https://pypi.tuna.tsinghua.edu.cn/simple
#matplotlib：基础的绘图库。几乎什么图都能画，但代码稍显繁琐。
#seaborn：基于 Matplotlib 的高级绘图库，专门做统计图形。几行代码就能画出漂亮的散点图、热力图，非常适合分析数据分布。
#graphviz：专门用来画树形结构的工具。XGBoost 本质是很多决策树，用它可以画出单棵树的形状，帮助理解模型。

pip install shap optuna 
#SHAP用于解释 XGBoost 模型的输出，帮你理解哪些特征导致了“风险”上升。
#optuna可选，自动超参数调优，比网格搜索更高效

pip install jupyter notebook   -i https://pypi.tuna.tsinghua.edu.cn/simple
#jupyter notebook可选，交互式开发环境

国内镜像源地址
https://pypi.tuna.tsinghua.edu.cn/simple
https://mirrors.aliyun.com/pypi/simple/
https://pypi.mirrors.ustc.edu.cn/simple/
```

## 四、验证环境
```bash
import xgboost as xgb
import sklearn
import shap
print(f"XGBoost version: {xgb.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
```
## 启动 Jupyter Notebook
jupyter notebook