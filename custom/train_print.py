"""
训练与结果绘图模块

本模块包含指定参数进行训练，处理训练结果并绘制折线图的功能。
主要功能包括：
- 构建命令行参数： 根据提供的预训练权重文件路径、模型配置文件路径、数据集路径、训练轮数和批次大小，构建了用于执行训练的命令行命令
- 使用subprocess.run()方法执行构建好的命令行命令，启动训练过程
- 读取存储训练结果数据的CSV文件
- 将训练过程中的性能指标变化趋势绘制成图

作者:
创建日期: 2024/3/15
"""
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import params


# 训练
def train(config):
    """
        功能：
        - 构建命令行参数进行训练

        参数：
        - config (yaml文件): 包含:
                                - weights:   # str: 预训练权重文件路径
                                - cfg:   # str: 模型配置文件路径
                                - data:   # str: 数据集路径
                                - epochs:   # int: 训练轮数
                                - batch_size:   # int: 批次大小

        返回：
        - 命令行参数
        """
    command = [
        "python", config['train_script'],
        "--weights", config['weights'],
        "--cfg", config['cfg'],
        "--data", config['data'],
        "--epochs", str(config['epochs']),
        "--batch-size", str(config['batch_size'])
    ]
    subprocess.run(command)


# 绘图
def print_results(config):
    """
           功能：
           - 根据训练结果绘制折线图

           参数：
           - config (yaml文件): 包含:
                                   - path:   # str: 训练结果文件路径
                                   - y_column:   # str: 模型配置文件路径
                                   - title:   # str: y轴的值
                                   - xlabel:   # str: x轴的名字
                                   - ylabel:   # str: y轴的名字

           返回：
           - 无
           """
    plt.figure(figsize=(10, 6))
    data = pd.read_csv(config['path'])
    x = data['               epoch']
    y = data[config['y_column']]
    plt.plot(x, y, label=config['path'])
    plt.title(config['title'])
    plt.xlabel(config['xlabel'])
    plt.ylabel(config['ylabel'])
    plt.show()


# 主函数
def main():
    train(params.train_params)
    print_results(params.print_params)


if __name__ == '__main__':
    main()




