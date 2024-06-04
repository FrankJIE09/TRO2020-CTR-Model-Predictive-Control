import numpy as np
from Tube import Tube
from CTR_MPC import CTR_MPC
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from matplotlib import cm
import matplotlib.pyplot as plt
import time

# 定义每根管子的参数，从最内层管子开始编号
# 参数依次是：长度、弯曲长度、内径、外径、刚度、扭转刚度、X方向弯曲、Y方向弯曲
tube1 = Tube(400e-3, 200e-3, 2 * 0.35e-3, 2 * 0.55e-3, 70.0e+9, 10.0e+9, 12, 0)
tube2 = Tube(300e-3, 150e-3, 2 * 0.7e-3, 2 * 0.9e-3, 70.0e+9, 10.0e+9, 6, 0)

# 初始关节变量的猜测
q = np.array([0.0, 0.0, 0, 0.0])

# 关节的初始位置
q_init = np.array([-300e-3, -200e-3, 0, 0])

# 初始扭转（用于常微分方程求解器）
uz_0 = np.array([0.0, 0.0, 0.0])
u1_xy_0 = np.array([[0.0], [0.0]])

# 时间数组，从0到2，分为10个点
t = np.linspace(0, 2, num=10)

# 速度（米/秒）
v = 1e-3

# 初始化关节变量数组，用于存储结果
q_array = np.zeros((1, 4))

# 初始化期望位置数组
x_d_array = np.array([[0e-2], [-3.32e-02], [9.21e-02]])

# 迭代计算
for i in t:
    # 计算期望位置
    x_d = np.array([[0 + v * i], [-3.32e-02 + v * i], [9.21e-02]])

    # 创建CTR_MPC对象
    CTR = CTR_MPC(tube1, tube2, q, q_init, x_d, 0.01)

    # 最小化函数，更新关节变量
    q = CTR.minimize(q_init, q)

    # 更新期望位置数组
    x_d_array = np.concatenate((x_d_array, x_d), axis=0)

    # 更新关节变量数组
    q_array = np.concatenate((q_array, q.reshape(1, 4)), axis=0)

# 绘制机器人的形状
fig = plt.figure()
ax = plt.axes(projection='3d')
counter = 1

# 绘制不同时间点的机器人形状
for i in t:
    # 计算期望位置
    x_d = np.array([[0 + v * i], [-3.32e-02 + v * i], [9.21e-02]])

    # 从关节变量数组中获取当前关节变量
    q = q_array[counter, :].reshape(4, )

    # 通过常微分方程求解器计算机器人的位置
    CTR.ode_solver(q)

    # 绘制期望位置的红色点
    ax.scatter(x_d[0, 0], x_d[1, 0], x_d[2, 0], c='r', marker='o')

    # 绘制机器人形状的蓝色线
    ax.plot(CTR.r[:, 0], CTR.r[:, 1], CTR.r[:, 2], '-b')

    # 自动缩放坐标轴
    ax.auto_scale_xyz([np.amin(CTR.r[:, 0]), np.amax(CTR.r[:, 0]) + 0.01],
                      [np.amin(CTR.r[:, 1]), np.amax(CTR.r[:, 1]) + 0.01],
                      [np.amin(CTR.r[:, 2]), np.amax(CTR.r[:, 2]) + 0.01])

    counter += 1

# 绘制最后的期望位置点，并添加图例
a = ax.scatter(x_d[0, 0], x_d[1, 0], x_d[2, 0], c='r', marker='o', label='Desired Position')
leg = ax.legend()

# 设置坐标轴标签
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')

# 添加网格
plt.grid(True)

# 显示图形
plt.show()
