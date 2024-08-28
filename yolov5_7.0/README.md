# YOLOV5使用

- [ ] ❓`WARNING: Ignore distutils configs in setup.cfg due to encoding errors`

- [x] `UnicodeDecodeError: 'gbk' codec can't decode byte 0x80 in position 268: illegal multibyte sequence`

  > 需要保证requirements.txt文件不存在中文；

- [x] 安装requirements.txt。

  > 1. pip install -r requirements.txt -i https://mirrors.huaweicloud.com/repository/pypi/simple
  > 2. ipython：python3.9的时候需要8.13；python3.8需要8.1~8.12；

- [x] clearml如果希望使用，除了pip安装库以外，需要在官网配置。

  在命令行中输入cleanml_init,将某个api字典复制；

  [ClearML入门：简化机器学习解决方案的开发和管理-CSDN博客](https://blog.csdn.net/qq_40243750/article/details/126445671)

- [x] clearML和COMET在前期需要在loggers的初始化函数中，重新设置LOGGERS元组，去除clearML和comet；

- [x] `DeprecationWarning: Please use `spmatrix` from the `scipy.sparse` namespace, the `scipy.sparse.base` namespace is deprecated.`

  scipy和numpy在requirements.txt中的>=变成==；对应地，tensorflow和matplotlib会对应地选择版本；

  **最好通过百度，搜索，而不要通过pip自行搜索，这是漫无目的的**

- [ ] ``np.float` was a deprecated alias for the builtin `float`. To avoid this error in existing code, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.` 

  可以选择降低版本到1.20以内的；但是最好选择自行修改；

- [ ] train.py文件中训练参数的说明

  [YOLOV5训练代码train.py训练参数解析_yolov5 train.py参数-CSDN博客](https://blog.csdn.net/m0_47026232/article/details/129869740)

- [ ] 超参数优化文件：data/hyps

  [YOLO5的数据增强和权重设置hyp.scratch-med.yaml文件解读，degrees角度旋转和水平、垂直翻转解释-CSDN博客](https://blog.csdn.net/qq_51570094/article/details/124350214)

  >1. voc.yaml: 在voc上进化；
  >2. object365：在objects365上优化；
  >3. low: 小规模模型 NS；
  >4. med：中规模，M；
  >5. high: 大规模：LX等

- [ ] better comments插件的使用：[vscode 插件-better comments-代码注释高亮 - suwanbin - 博客园 (cnblogs.com)](https://www.cnblogs.com/suwanbin/p/13263732.html)
- [ ] 
