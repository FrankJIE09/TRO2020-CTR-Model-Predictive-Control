'''
#### 主要类和方法

- **CTR_MPC类**: 主要用于控制连续管状机器人（CTR）的模型预测控制（MPC）。
  - **__init__**: 初始化管子参数和状态变量。
  - **reset**: 重置管子状态。
  - **ode_eq**: 定义CTR的常微分方程。
  - **ode_solver**: 解决CTR的常微分方程，更新机器人状态。
  - **cost**: 计算成本函数。
  - **jac**: 计算成本函数的雅可比矩阵。
  - **inequality**: 定义关节变量的不等式约束。
  - **fun_hess_cost**: 计算拉格朗日的Hessian矩阵。
  - **fun_der_inequality**: 计算不等式约束的梯度。
  - **fun_der**: 计算成本函数的梯度。
  - **kk**: 计算KKT条件。
  - **mat**: 构建优化问题的矩阵。
  - **backtrack**: 计算步长。
  - **minimize**: 解决MPC问题。
  - **solve_mpc**: 通过KKT条件解决MPC问题。

- **main函数**: 定义管子参数，初始化MPC控制器，解决并绘制CTR的状态和期望位置。

这个代码示例展示了如何使用MPC算法来控制连续管状机器人的运动，并通过求解常微分方程来更新机器人的状态。
'''



import numpy as np
from scipy.integrate import solve_ivp
from Segment import Segment
from Tube import Tube
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import minimize
import time
from numdifftools import Jacobian, Hessian

class CTR_MPC:
    def __init__(self, tube1, tube2, q, q_0, x_d, Tol):  # 方法 1,2 ---> 边值问题和初值问题
        self.accuracy = Tol
        self.eps = 1e-5
        self.q_0 = q_0
        self.x_d = x_d
        self.q = q
        self.tube1, self.tube2, self.q, self.q_0 = tube1, tube2, q.astype(float), q_0.astype(float)
        # 管子基座位置（从模板，即s=0开始）
        self.beta = q[0:2] + self.q_0[0:2]
        self.segment = Segment(self.tube1, self.tube2, self.beta)

        self.span = np.append([0], self.segment.S)
        self.r_0 = np.array([0, 0, 0]).reshape(3, 1)  # 机器人的初始位置
        self.alpha_1_0 = self.q[2] + self.q_0[2]  # 管子1的初始扭转角
        self.R_0 = np.array(
            [[np.cos(self.alpha_1_0), -np.sin(self.alpha_1_0), 0], [np.sin(self.alpha_1_0), np.cos(self.alpha_1_0), 0],
             [0, 0, 1]]) \
            .reshape(9, 1)  # 初始旋转矩阵
        self.alpha_0 = q[2:].reshape(2, 1) + q_0[2:].reshape(2, 1) - self.alpha_1_0  # 所有管子的初始扭转角
        self.Length = np.empty(0)
        self.r = np.empty((0, 3))
        self.u_z = np.empty((0, 2))
        self.alpha = np.empty((0, 2))

    def reset(self, q):
        self.q = q
        self.beta = q[0:2] + self.q_0[0:2]
        self.segment = Segment(self.tube1, self.tube2, self.beta)

        self.span = np.append([0], self.segment.S)
        self.r_0 = np.array([0, 0, 0]).reshape(3, 1)  # 机器人的初始位置
        self.alpha_1_0 = q[2] + self.q_0[2]  # 管子1的初始扭转角
        self.R_0 = np.array(
            [[np.cos(self.alpha_1_0), -np.sin(self.alpha_1_0), 0], [np.sin(self.alpha_1_0), np.cos(self.alpha_1_0), 0],
             [0, 0, 1]]) \
            .reshape(9, 1)  # 初始旋转矩阵
        self.alpha_0 = q[2:].reshape(2, 1) + self.q_0[2:].reshape(2, 1) - self.alpha_1_0  # 所有管子的初始扭转角
        self.Length = np.empty(0)
        self.r = np.empty((0, 3))
        self.u_z = np.empty((0, 2))
        self.alpha = np.empty((0, 2))

    # 两根管子的CTR的常微分方程
    def ode_eq(self, s, y, ux_0, uy_0, ei, gj):
        # y的第一个元素是第一根管子沿x方向的曲率，
        # y的第二个元素是第一根管子沿y方向的曲率
        # 接下来的两个元素是沿z方向的曲率，例如，y=[ u1_z  u2_z]
        # 接下来的两个元素是扭转角，alpha_i
        # 最后12个元素分别是r（位置）和R（方向）
        dydt = np.empty([18, 1])
        tet1 = y[4]
        tet2 = y[5]
        R_tet1 = np.array([[np.cos(tet1), -np.sin(tet1), 0], [np.sin(tet1), np.cos(tet1), 0],
                           [0, 0, 1]])
        R_tet2 = np.array([[np.cos(tet2), -np.sin(tet2), 0], [np.sin(tet2), np.cos(tet2), 0],
                           [0, 0, 1]])
        u2 = R_tet2.transpose() @ np.array([[y[0]], [y[1]], [y[2]]]) + dydt[
            4] * np.array([[0], [0], [1]])  # 管子2的曲率向量
        u = np.array([y[0], y[1], y[2], u2[0, 0], u2[1, 0], y[3]])
        u1 = np.array([y[0], y[1], y[2]]).reshape(3, 1)

        # 估算扭转曲率和扭转角
        for i in np.argwhere(gj != 0):
            dydt[2 + i] = ((ei[i]) / (gj[i])) * (u[i * 3] * uy_0[i] - u[i * 3 + 1] * ux_0[i])  # ui_z
            dydt[4 + i] = y[2 + i] - y[2]  # alpha_i

        # 估算第一根管子沿x和y方向的曲率
        K_inv = np.diag(np.array([1 / np.sum(ei), 1 / np.sum(ei), 1 / np.sum(gj)]))
        K1 = np.diag(np.array([ei[0], ei[0], gj[0]]))
        K2 = np.diag(np.array([ei[1], ei[1], gj[1]]))
        dR_tet1 = np.array([[-np.sin(tet1), -np.cos(tet1), 0], [np.cos(tet1), -np.sin(tet1), 0],
                            [0, 0, 1]])
        dR_tet2 = np.array([[-np.sin(tet2), -np.cos(tet2), 0], [np.cos(tet2), -np.sin(tet2), 0],
                            [0, 0, 1]])
        u_hat1 = np.array([[0, -u1[2][0], u1[1][0]], [u1[2][0], 0, -u1[0][0]], [-u1[1][0], u1[0][0], 0]], dtype=np.float64)
        u_hat2 = np.array([[0, -u2[2][0], u2[1][0]], [u2[2][0], 0, -u2[0][0]], [-u2[1][0], u2[0][0], 0]], dtype=np.float64)
        u_s1 = np.array([ux_0[0], uy_0[0], 0]).reshape(3, 1)
        u_s2 = np.array([ux_0[1], uy_0[1], 0]).reshape(3, 1)
        du = np.zeros((3, 1), dtype=np.float64)
        du = -K_inv @ (R_tet1 @ (K1 @ (dydt[4] * dR_tet1.transpose() @ u1) + u_hat1 @ K1 @ (u1 - u_s1)) + R_tet2 @ (
                K2 @ (dydt[5] * dR_tet2.transpose() @ u2) + u_hat2 @ K2 @ (u2 - u_s2)))
        dydt[0] = du[0, 0]
        dydt[1] = du[1, 0]
        R = np.array(
            [[y[9], y[10], y[11]], [y[12], y[13], y[14]], [y[15], y[16], y[17]]])  # 管子1的旋转矩阵

        # 估算R和r
        e3 = np.array([[0.0], [0.0], [1.0]])
        dr = R @ e3
        dR = (R @ u_hat1).ravel()

        dydt[6] = dr[0, 0]
        dydt[7] = dr[1, 0]
        dydt[8] = dr[2, 0]

        for k in range(3, 12):
            dydt[6 + k] = dR[k - 3]
        return dydt.ravel()

    def ode_solver(self, q):
        self.reset(q)
        u1_xy_0 = np.array([[0.0], [0.0]])
        u1_xy_0[0, 0] = (1 / (self.segment.EI[0, 0] + self.segment.EI[1, 0])) * \
                        (self.segment.EI[0, 0] * self.segment.U_x[0, 0] +
                         self.segment.EI[1, 0] * self.segment.U_x[1, 0] * np.cos(- self.alpha_0[1, 0]) +
                         self.segment.EI[1, 0] *
                         self.segment.U_y[1, 0] * np.sin(- self.alpha_0[1, 0]))
        u1_xy_0[1, 0] = (1 / (self.segment.EI[0, 0] + self.segment.EI[1, 0])) * \
                        (self.segment.EI[0, 0] * self.segment.U_y[0, 0] +
                         -self.segment.EI[1, 0] * self.segment.U_x[1, 0] * np.sin(- self.alpha_0[1, 0]) +
                         self.segment.EI[1, 0] *
                         self.segment.U_y[1, 0] * np.cos(- self.alpha_0[1, 0]))
        uz_0 = np.array([0.0, 0.0]).reshape(2, 1)

        # 重置常微分方程求解器的初始参数
        for seg in range(0, len(self.segment.S)):
            # 初始条件：管子1的3个初始曲率，管子2和3的3个初始扭转，3个初始角度，
            # 3个初始位置，9个初始旋转矩阵
            y_0 = np.vstack((u1_xy_0, uz_0, self.alpha_0, self.r_0, self.R_0)).ravel()
            s = solve_ivp(lambda s, y: self.ode_eq(s, y, self.segment.U_x[:, seg], self.segment.U_y[:, seg],
                                                   self.segment.EI[:, seg], self.segment.GJ[:, seg]),
                          [self.span[seg], self.span[seg + 1]],
                          y_0, method='RK23', max_step=self.accuracy)
            self.Length = np.append(self.Length, s.t)
            ans = s.y.transpose()
            self.u_z = np.vstack((self.u_z, ans[:, (2, 3)]))
            self.alpha = np.vstack((self.alpha, ans[:, (4, 5)]))
            self.r = np.vstack((self.r, ans[:, (6, 7, 8)]))
            dtheta2 = ans[-1, 3] - ans[-1, 2]
            # 为下一个段估算新的边界条件
            uz_0 = self.u_z[-1, :].reshape(2, 1)
            self.r_0 = self.r[-1, :].reshape(3, 1)
            self.R_0 = np.array(ans[-1, 9:]).reshape(9, 1)
            self.alpha_0 = self.alpha[-1, :].reshape(2, 1)
            u1 = ans[-1, (0, 1, 2)].reshape(3, 1)
            if seg < len(
                    self.segment.S) - 1:  # 强制弯矩连续性以估算下一个段的初始曲率

                K1 = np.diag(np.array([self.segment.EI[0, seg], self.segment.EI[0, seg], self.segment.GJ[0, seg]]))
                K2 = np.diag(np.array([self.segment.EI[1, seg], self.segment.EI[1, seg], self.segment.GJ[1, seg]]))
                U1 = np.array([self.segment.U_x[0, seg], self.segment.U_y[0, seg], 0]).reshape(3, 1)
                U2 = np.array([self.segment.U_x[1, seg], self.segment.U_y[1, seg], 0]).reshape(3, 1)

                GJ = self.segment.GJ
                GJ[self.segment.EI[:, seg + 1] == 0] = 0
                K1_new = np.diag(
                    np.array([self.segment.EI[0, seg + 1], self.segment.EI[0, seg + 1], self.segment.GJ[0, seg + 1]]))
                K2_new = np.diag(
                    np.array([self.segment.EI[1, seg + 1], self.segment.EI[1, seg + 1], self.segment.GJ[1, seg + 1]]))
                U1_new = np.array([self.segment.U_x[0, seg + 1], self.segment.U_y[0, seg + 1], 0]).reshape(3, 1)
                U2_new = np.array([self.segment.U_x[1, seg + 1], self.segment.U_y[1, seg + 1], 0]).reshape(3, 1)

                R_theta2 = np.array(
                    [[np.cos(self.alpha_0[1, 0]), -np.sin(self.alpha_0[1, 0]), 0],
                     [np.sin(self.alpha_0[1, 0]), np.cos(self.alpha_0[1, 0]), 0],
                     [0, 0, 1]])
                e3 = np.array([0, 0, 1]).reshape(3, 1)
                u2 = R_theta2.transpose() @ u1 + dtheta2 * e3
                K_inv_new = np.diag(np.array(
                    [1 / (self.segment.EI[0, seg + 1] + self.segment.EI[1, seg + 1]),
                     1 / (self.segment.EI[0, seg + 1] + self.segment.EI[1, seg + 1]),
                     1 / (self.segment.GJ[0, seg + 1] + self.segment.GJ[1, seg + 1])]))
                u1_new = K_inv_new @ (K1 @ (u1 - U1) + R_theta2 @ K2 @ (u2 - U2) + K1_new @ U1_new
                                      + R_theta2 @ K2_new @ U2_new - R_theta2 @ K2_new @ (dtheta2 * e3))
                u1_xy_0 = u1_new[0:2, 0].reshape(2, 1)
        return

    # 成本函数，输入是关节变量q，拉格朗日参数w，松弛变量s，期望位置x_d
    def cost(self, q):
        self.ode_solver(q)
        Error = 1000 * (self.r[-1, :].reshape(3, 1) - self.x_d.reshape(3, 1))
        C = np.sum((Error.T @ Error), axis=0)
        return C

    # 估算成本函数的雅可比矩阵
    def jac(self, q):
        jac = np.zeros((4,))
        r = self.cost(q)
        for i in range(0, 4):
            q[i] = q[i] + self.eps
            r_perturb = (self.cost(q) - r) / self.eps
            jac[i] = r_perturb.reshape(1, )
            q[i] = q[i] - self.eps
        return jac

    # 关节变量的不等式约束
    def inequality(self, q, q0):
        I = np.array([[-q[0] - q0[0]], [-q[1] - q0[1]], [-q[0] - q0[0] + q[1] + q0[1]], [q[0] + q0[0] + 0.4]])
        return I

    # 拉格朗日的Hessian矩阵
    def fun_hess_cost(self, q, w, s, x_d, q0):
        return Hessian(lambda q: self.cost(q, w, s, x_d, q0), step=self.eps)(q)

    # 成本函数的梯度
    def fun_der_inequality(self, q, q0):
        return Jacobian(lambda q: self.inequality(q, q0), step=self.eps)(q)

    # 成本函数的梯度
    def fun_der(self, q, x_d):
        dF = optimize.approx_fprime(q, self.ode_solver, self.eps, x_d)
        return dF.reshape(4, 1)

    def kk(self, q, q0, x_d, w, s, mu):
        S = np.diag(1 / s[:, 0])
        e = np.array([[1], [1], [1], [1]])
        k1 = self.fun_der(q, x_d) - (self.fun_der_inequality(q, q0)).T @ w
        k2 = w - mu * S @ e
        k3 = -self.inequality(q, q0) + s
        k = np.concatenate((k1, k2, k3), axis=0)
        return k

    def mat(self, q, q0, x_d, w, s):
        S = np.diag(1 / s[:, 0])
        e = np.array([[1], [1], [1], [1]])
        Sig = S * np.diag(w[:, 0])
        m1 = np.concatenate((self.fun_hess_cost(q, w, s, x_d, q0), np.zeros((4, 4)),
                             -(self.fun_der_inequality(q, q0)).T), axis=1)
        m2 = np.concatenate((np.zeros((4, 4)), Sig, np.eye(4)), axis=1)
        m3 = np.concatenate((-self.fun_der_inequality(q, q0), np.eye(4), np.zeros((4, 4))), axis=1)
        m = np.concatenate((m1, m2, m3), axis=0)
        return m

    def backtrack(self, s, ds, alpha):
        Tau = 0.995
        counter = 1
        while np.all(alpha * ds + Tau * s <= 0) and counter < 20:
            alpha *= 0.9
            counter += 1
        return alpha

    # 解决MPC问题
    def minimize(self, q_init, q):
        eps = 0.0001
        # 限制条件
        ineq_cons = {'type': 'ineq',
                     'fun': lambda q: np.array([-q[0] - q_init[0] - eps,
                                                -q[1] - q_init[1] - eps,
                                                q[0] + q_init[0] + 0.4 - eps,
                                                q[1] + q_init[1] + 0.3,
                                                q_init[1] - q_init[0] - eps - q[0] + q[1]]),
                     'jac': lambda q: np.array([[-1.0, 0, 0, 0],
                                                [0, -1.0, 0, 0],
                                                [1.0, 0.0, 0, 0],
                                                [0.0, 1.0, 0, 0],
                                                [-1.0, 1.0, 0, 0]])}

        res = minimize(self.cost, q, method='SLSQP', jac=self.jac,
                       constraints=ineq_cons, options={'ftol': 0.75e-3})

        # print(res.x)
        return res.x

    def solve_mpc(self, q, q0, x_d, w, s, mu):
        K = self.kk(q, q0, x_d, w, s, mu)
        counter = 1
        cost = self.cost(q)
        Q = q.reshape(1, 4)
        C = np.array([cost]).reshape(1, 1)
        while np.linalg.norm(K) >= 0.1 and cost >= 2 and counter <= 50:
            alpha_0 = 1
            M = self.mat(q, q0, x_d, w, s)
            delta = np.linalg.solve(M, -K)
            dq = delta[0:4, :]
            ds = delta[4:8, :]
            dw = delta[8:12, :]
            alpha_s = self.backtrack(s, ds, alpha_0)
            alpha_w = self.backtrack(w, dw, alpha_0)
            q += (alpha_s * dq).reshape(4, )
            cost_new = self.cost(q)
            counter2 = 1
            while cost_new > cost and counter2 <= 50:
                q -= (alpha_s * dq).reshape(4, )
                alpha_0 = 0.5 * alpha_0
                alpha_s = self.backtrack(s, ds, alpha_0)
                q += (alpha_s * dq).reshape(4, )
                cost_new = self.cost(q)
                counter2 += 1
            self.ode_solver(q)
            s += alpha_s * ds
            w += alpha_w * dw
            mu *= 0.98
            K = self.kk(q, q0, x_d, w, s, mu)
            counter += 1
            #print(counter)
            #print(self.ode_solver(q))
            C = np.concatenate((C, np.array([cost]).reshape(1, 1)), axis=0)
            Q = np.concatenate((Q, q.reshape(1, 4)), axis=0)
        return Q, C

def main():
    start_time = time.time()

    # 定义每根管子的参数，从最内层管子开始编号
    # 参数依次是：长度、弯曲长度、内径、外径、刚度、扭转刚度、X方向弯曲、Y方向弯曲
    tube1 = Tube(400e-3, 200e-3, 2 * 0.35e-3, 2 * 0.55e-3, 70.0e+9, 10.0e+9, 12, 0)
    tube2 = Tube(300e-3, 150e-3, 2 * 0.7e-3, 2 * 0.9e-3, 70.0e+9, 10.0e+9, 6, 0)
    # 关节变量
    q = np.array([0.0, 0.0, 0.0, 0.0])
    # 关节的初始位置
    q_init = np.array([-300e-3, -200e-3, 0, 0])
    # 初始扭转（用于ivp求解器）
    uz_0 = np.array([0.0, 0.0, 0.0])
    u1_xy_0 = np.array([[0.0], [0.0]])
    # 期望位置 -7.45336254e-02  1.19058021e-01
    x_d = np.array([[0 + 1e-3], [-3.32759076e-02 + 0.001], [9.21576565e-02]])
    CTR = CTR_MPC(tube1, tube2, q, q_init, x_d, 0.01)
    CTR.ode_solver(q)
    print(CTR.r[-1, :])
    CTR.minimize(q_init, q)
    print(CTR.r[-1, :])

    # 绘制机器人的形状
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(CTR.r[:, 0], CTR.r[:, 1], CTR.r[:, 2], '-b', label='CTR Robot')
    ax.scatter(x_d[0, 0], x_d[1, 0], x_d[2, 0], c='r', marker='o', label='desired')
    ax.auto_scale_xyz([np.amin(CTR.r[:, 0]), np.amax(CTR.r[:, 0]) + 0.01],
                      [np.amin(CTR.r[:, 1]), np.amax(CTR.r[:, 1]) + 0.01],
                      [np.amin(CTR.r[:, 2]), np.amax(CTR.r[:, 2]) + 0.01])
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    plt.grid(True)
    plt.legend()
    plt.show()

    return

if __name__ == "__main__":
    main()
