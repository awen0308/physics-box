# 物理盒子

这是一个用 Godot 4 做的小游戏。屏幕里有个小球，会受重力往下掉，撞到墙会弹起来。

好玩的地方在后面。我用 Python 写了一个神经网络，让它看着小球怎么动，学会预测小球下一步会去哪里。

游戏里能看到三种颜色：
青色的线是小球真实走过的路
粉色的小点是模型猜的小球下一步位置
绿色的线是模型想象的小球接下来 30 步会怎么走

如果按 M 键，游戏会关掉真实的物理引擎，改由模型来推动小球。也就是说，模型已经把物理规律学进去了。

## 怎么玩

1. 用 Godot 4 打开这个项目，运行 main.tscn
2. 方向键或者 W A D 控制小球，鼠标左键能把小球吸过来，右键可以设一个目标点
3. 玩个几分钟之后按 S 把数据存下来
4. 在电脑的终端里运行 python training/train.py 训练模型
5. 回到游戏按 L 加载模型，这时候就能看到粉点和绿线了

## 文件都是干嘛的

scripts 文件夹里是游戏的逻辑，main.gd 是主程序，data_collector.gd 负责记录小球的数据，world_model.gd 负责把训练好的模型加载进来做推理。

training 文件夹里是 Python 的训练代码。mlp.py 是神经网络，train.py 用来训练，rnn.py 是另一个更高级的版本，evaluate.py 用来算模型准不准。

data 文件夹里是你在游戏里采集的数据，models 文件夹里是训练完的模型。

## 用到的东西

Godot 4 写游戏，Python 写神经网络，神经网络只用 NumPy 实现，没有用 TensorFlow 或者 PyTorch 这种框架。

## 训练

先把依赖装上：

pip install -r requirements.txt

然后跑：

python training/train.py

训练好的模型会存到 models/world_model.json。

## 说明

这个项目是我自己想的点子和整体设计，开发过程中借助了 AI 工具帮忙查资料和调代码。
