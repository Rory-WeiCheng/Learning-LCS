import pdb
import time
import casadi
import numpy as np
from casadi import *
from scipy import interpolate
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


# 2023.4.24 fix the notation issue, follow Alp's work
class LCS_Gen:

    def __init__(self, n_state, n_control, n_lam, A=None, B=None, D=None, dyn_offset=None,
                 E=None, H=None, F=None, lcp_offset=None):
        self.n_state = n_state
        self.n_control = n_control
        self.n_lam = n_lam

        self.A = DM(A)
        self.B = DM(B)
        self.D = DM(D)
        self.dyn_offset = DM(dyn_offset)
        self.E = DM(E)
        self.H = DM(H)
        self.F = DM(F)
        self.lcp_offset = DM(lcp_offset)

        # define the system variable
        x = SX.sym('x', self.n_state)
        u = SX.sym('u', self.n_control)
        xu_pair = vertcat(x, u)
        lam = SX.sym('lam', self.n_lam)
        x_next = SX.sym('x_next', self.n_state)

        # define the model system matrices, M stands for model
        Model_para = []
        A_M = SX.sym('A_M', self.n_state, self.n_state)
        Model_para += [vec(A_M)]
        B_M = SX.sym('B_M', self.n_state, self.n_control)
        Model_para += [vec(B_M)]
        D_M = SX.sym('D_M', self.n_state, self.n_lam)
        Model_para += [vec(D_M)]
        dyn_offset_M = SX.sym('dyn_offset_M', self.n_state)
        Model_para += [vec(dyn_offset_M)]
        E_M = SX.sym('E_M', self.n_lam, self.n_state)
        Model_para += [vec(E_M)]
        H_M = SX.sym('H_M', self.n_lam, self.n_control)
        Model_para += [vec(H_M)]
        F_M = SX.sym('F_M', self.n_lam, self.n_lam)
        Model_para += [vec(F_M)]
        lcp_offset_M = SX.sym('lcp_offset_M', self.n_lam)
        Model_para += [vec(lcp_offset_M)]

        theta_M = vcat(Model_para)
        self.Form_theta_M = Function('Form_theta_M', [A_M, B_M, D_M, dyn_offset_M, E_M, H_M, F_M, lcp_offset_M],
                                     [theta_M])

        xu_theta_pair = vertcat(xu_pair, theta_M)

        # dynamics
        dyn = (self.A + A_M) @ x + (self.B + B_M) @ u + (self.D + D_M) @ lam + (self.dyn_offset + dyn_offset_M)
        self.dyn_fn = Function('dyn_fn', [xu_theta_pair, lam], [dyn])

        # lcp function
        lcp_cstr = (self.E + E_M) @ x + (self.H + H_M) @ u + (self.F + F_M) @ lam + (self.lcp_offset + lcp_offset_M)
        lcp_loss = dot(lam, lcp_cstr)
        self.lcp_loss_fn = Function('dyn_loss_fn', [xu_theta_pair, lam], [lcp_loss])
        self.lcp_cstr_fn = Function('dis_cstr_fn', [xu_theta_pair, lam], [lcp_cstr])

        # establish the qp solver to solve for lcp
        xu_theta = vertcat(xu_theta_pair)
        quadprog = {'x': lam, 'f': lcp_loss, 'g': lcp_cstr, 'p': xu_theta}
        opts = {'print_time': 0, 'osqp': {'verbose': False}}
        self.lcpSolver = qpsol('lcpSolver', 'osqp', quadprog, opts)

    def nextState(self, x_batch, u_batch, batch_A, batch_B, batch_D, batch_dynamic_offset,
                  batch_E, batch_H, batch_F, batch_lcp_offset):
        batch_size = x_batch.shape[0]
        # theta_M_list = []
        # for i in range(batch_size):
        #     A_Mi = batch_A[i].flatten('F')
        #     B_Mi = batch_B[i].flatten('F')
        #     D_Mi = batch_D[i].flatten('F')
        #     dynamic_offset_Mi = batch_dynamic_offset[i]
        #     E_Mi = batch_E[i].flatten('F')
        #     H_Mi = batch_H[i].flatten('F')
        #     F_Mi = batch_F[i].flatten('F')
        #     lcp_offset_Mi = batch_lcp_offset[i]
        #     theta_Mi = np.concatenate([A_Mi, B_Mi, D_Mi, dynamic_offset_Mi, E_Mi, H_Mi, F_Mi, lcp_offset_Mi])
        #     theta_M_list.append(theta_Mi)
        #     # print(lcp_offset_Mi + self.lcp_offset)
        # theta_M_batch = np.stack(theta_M_list,axis=1).T
        theta_M_batch = self.Form_theta_M(batch_A, batch_B, batch_D, batch_dynamic_offset, batch_E, batch_H, batch_F,
                                          batch_lcp_offset).full().T
        xu_theta_batch = np.hstack((x_batch, u_batch, theta_M_batch))

        xu_theta_batch = DM(xu_theta_batch)
        sol = self.lcpSolver(p=xu_theta_batch.T, lbx=0., lbg=0.)
        lam_batch = sol['x'].full()

        # compute the next state batch
        x_next_batch = self.dyn_fn(xu_theta_batch.T, lam_batch).full()

        return x_next_batch.T, lam_batch.T

    # this only works for dim(x) is one
    def plotData(self, x_batch, x_next_batch, lam_batch=0):
        plt.scatter(x_batch, x_next_batch, s=1)
        plt.show()


# modify the lcs_class from wanxin's code, incorperate collected local LCS matricess as part of the data
# The dynamics for each data point is hence different
# 2023.4.24 fix the notation issue, follow Alp's work
class LCS_VN:

    def __init__(self, n_state, n_control, n_lam, A=None, B=None, D=None, dyn_offset=None,
                 E=None, H=None, G_para=None, S=None, lcp_offset=None, F_stiffness=1.0):
        self.n_state = n_state
        self.n_control = n_control
        self.n_lam = n_lam

        # define system matrices
        self.tunable_para = []
        if A is None:
            self.A = SX.sym('A', n_state, n_state)
            self.tunable_para += [vec(self.A)]
        else:
            self.A = DM(A)

        if B is None:
            self.B = SX.sym('B', n_state, n_control)
            self.tunable_para += [vec(self.B)]
        else:
            self.B = DM(B)

        if D is None:
            self.D = SX.sym('D', n_state, n_lam)
            self.tunable_para += [vec(self.D)]
        else:
            self.D = DM(D)

        if dyn_offset is None:
            self.dyn_offset = SX.sym('dyn_offset', n_state)
            self.tunable_para += [vec(self.dyn_offset)]
        else:
            self.dyn_offset = DM(dyn_offset)

        if E is not None:
            self.E = DM(E)
        else:
            self.E = SX.sym('E', n_lam, n_state)
            self.tunable_para += [vec(self.E)]

        if H is None:
            self.H = SX.sym('H', n_lam, n_control)
            self.tunable_para += [vec(self.H)]
        else:
            self.H = DM(H)

        if G_para is not None:
            self.G_para = DM(G_para)
            self.G = self.toMatG(G_para)
        else:
            self.G_para = SX.sym('G_para', int((self.n_lam + 1) * self.n_lam / 2))
            self.G = self.toMatG(self.G_para)
            self.tunable_para += [vec(self.G_para)]

        if S is None:
            self.S = SX.sym('S', n_lam, n_lam)
            self.tunable_para += [vec(self.S)]
        else:
            self.S = DM(S)

        if lcp_offset is None:
            self.lcp_offset = SX.sym('lcp_offset', n_lam)
            self.tunable_para += [vec(self.lcp_offset)]
        else:
            self.lcp_offset = DM(lcp_offset)

        self.theta = vcat(self.tunable_para)
        self.n_theta = self.theta.numel()

        self.F = self.G @ self.G.T + F_stiffness * np.eye(self.n_lam) + self.S - self.S.T

        self.A_fn = Function('A_fn', [self.theta], [self.A])
        self.B_fn = Function('B_fn', [self.theta], [self.B])
        self.D_fn = Function('D_fn', [self.theta], [self.D])
        self.dyn_offset_fn = Function('dyn_offset_fn', [self.theta], [self.dyn_offset])

        self.E_fn = Function('E_fn', [self.theta], [self.E])
        self.H_fn = Function('H_fn', [self.theta], [self.H])
        self.G_fn = Function('G_fn', [self.theta], [self.G])
        self.S_fn = Function('S_fn', [self.theta], [self.S])
        self.lcp_offset_fn = Function('lcp_offset_fn', [self.theta], [self.lcp_offset])

        self.F_fn = Function('F_fn', [self.theta], [self.F])

    def toMatG(self, G_para):

        if type(G_para) is casadi.SX:
            G = SX(self.n_lam, self.n_lam)
        else:
            G = DM(self.n_lam, self.n_lam)

        for j in range(self.n_lam):
            for i in range(j + 1):
                if i == j:
                    G[i, j] = (G_para[int((j + 1) * j / 2) + i])
                else:
                    G[i, j] = G_para[int((j + 1) * j / 2) + i]
        return G

    def diff(self, gamma=1e-2, epsilon=1., F_ref=0., w_F=0., D_ref=0., w_D=0.):

        # define the system variable
        x = SX.sym('x', self.n_state)
        u = SX.sym('u', self.n_control)
        x_next = SX.sym('x_next', self.n_state)

        # define the model system matrices, M stands for model
        Model_para = []
        A_M = SX.sym('A_M', self.n_state, self.n_state)
        Model_para += [vec(A_M)]
        B_M = SX.sym('B_M', self.n_state, self.n_control)
        Model_para += [vec(B_M)]
        D_M = SX.sym('D_M', self.n_state, self.n_lam)
        Model_para += [vec(D_M)]
        dyn_offset_M = SX.sym('dyn_offset_M', self.n_state)
        Model_para += [vec(dyn_offset_M)]

        E_M = SX.sym('E_M', self.n_lam, self.n_state)
        Model_para += [vec(E_M)]
        H_M = SX.sym('H_M', self.n_lam, self.n_control)
        Model_para += [vec(H_M)]
        F_M = SX.sym('F_M', self.n_lam, self.n_lam)
        Model_para += [vec(F_M)]
        lcp_offset_M = SX.sym('lcp_offset_M', self.n_lam)
        Model_para += [vec(lcp_offset_M)]

        # vectorize the matrices to be theta_M
        theta_M = vcat(Model_para)
        # define model parameter formation function
        self.Form_theta_M = Function('Form_theta_M', [A_M, B_M, D_M, dyn_offset_M, E_M, H_M, F_M, lcp_offset_M],
                                     [theta_M])
        n_theta_M = theta_M.numel()

        data_pair = vertcat(x, u, x_next)

        # define the variables for LCP
        lam = SX.sym('lam', self.n_lam)
        phi = SX.sym('phi', self.n_lam)

        # dynamics
        dyn = (self.A + A_M) @ x + (self.B + B_M) @ u + (self.D + D_M) @ lam + (self.dyn_offset + dyn_offset_M)
        dyn_loss = dot(dyn - x_next, dyn - x_next)

        # lcp loss
        dist = (self.E + E_M) @ x + (self.H + H_M) @ u + (self.F + F_M) @ lam + (self.lcp_offset + lcp_offset_M)
        lcp_aug_loss = dot(lam, phi) + 1 / gamma * dot(phi - dist, phi - dist)

        # define loss function
        total_loss = 1. * dyn_loss \
                     + 1. / epsilon * lcp_aug_loss \
                     + w_D * dot(vec(DM(D_ref)) - vec(self.D), vec(DM(D_ref)) - vec(self.D)) \
                     + w_F * dot(vec(DM(F_ref)) - vec(self.F), vec(DM(F_ref)) - vec(self.F))

        # establish the qp solver, now the data_theta should contain three parts:
        # 1. (x,u,x_next) 2. theta (learnable parameters) 3.theta_M (local state dependent lcs matrices input)
        data_theta = vertcat(data_pair, self.theta, theta_M)
        lam_phi = vertcat(lam, phi)
        quadprog = {'x': lam_phi, 'f': total_loss, 'p': data_theta}
        opts = {'print_time': 0, 'osqp': {'verbose': False}}
        self.qpSolver = qpsol('Solver_QP', 'osqp', quadprog, opts)
        # opts = {"print_time": 0, "printLevel": "none", "verbose": 0}
        # self.qpSolver = qpsol('Solver_QP', 'qpoases', quadprog, opts)
        # opts = {}
        # self.qpSolver = qpsol('Solver_QP', 'proxqp', quadprog, opts)

        # compute the jacobian from lam to theta
        mu = SX.sym('mu', self.n_lam + self.n_lam)
        L = total_loss + dot(mu, lam_phi)

        # compute the derivative of the value function with respect to theta using method 2
        dyn_loss_plus = dyn_loss + \
                        w_D * dot(vec(DM(D_ref)) - vec(self.D), vec(DM(D_ref)) - vec(self.D)) + \
                        w_F * dot(vec(DM(F_ref)) - vec(self.F), vec(DM(F_ref)) - vec(self.F))
        self.loss_fn = Function('loss_fn', [data_pair, lam_phi, mu, self.theta, theta_M],
                                [jacobian(L, self.theta).T, dyn_loss_plus, lcp_aug_loss])

        # define the dyn prediction, pred_xu_theta should also contain theta_M
        pred_xu_theta = vertcat(x, u, self.theta, theta_M)
        pred_loss = 1. * dyn_loss + 1. / epsilon * lcp_aug_loss
        pred_x = vertcat(x_next, lam, phi)
        pred_g = vertcat(lam, phi)
        pred_quadprog = {'x': pred_x, 'f': pred_loss, 'g': pred_g, 'p': pred_xu_theta}
        opts = {'print_time': 0, 'osqp': {'verbose': False}}
        self.pred_qpSolver = qpsol('pred_qpSolver', 'osqp', pred_quadprog, opts)
        self.pred_error_fn = Function('pred_error_fn', [x, x_next], [dot(x - x_next, x - x_next)])

    def step(self, batch_x, batch_u, batch_x_next, current_theta, batch_A, batch_B, batch_D, batch_dynamic_offset,
             batch_E, batch_H, batch_F, batch_lcp_offset):

        start_step_time = time.time()
        # do data processing for batch calculation
        # batch x,u,x_next, horizontal concatenate, then use transpose for input
        batch_size = batch_x.shape[0]
        data_batch = hcat((batch_x, batch_u, batch_x_next))
        # theta and theta_M
        theta_batch = repmat(DM(current_theta).T, batch_size, 1)


        theta_M_batch = self.Form_theta_M(batch_A, batch_B, batch_D, batch_dynamic_offset, batch_E, batch_H, batch_F,
                                          batch_lcp_offset).T

        end_data_time = time.time()
        print("Data_time: " + str(end_data_time - start_step_time))

        # Assemble data (x,u,x_next) and theta (learnt + data input)
        # print(data_batch.shape, theta_M_batch.shape, theta_batch.shape)
        data_theta_batch = hcat((data_batch, theta_batch, theta_M_batch))
        # data_theta_batch = np.hstack((batch_x, batch_u, batch_x_next, theta_batch, theta_M_batch))
        # print(time.time()-start_step_time)

        # establish the solver, solve for lambda and phi using QP and prepare to feed to the gradient step
        # start_QP_time = time.time()
        sol = self.qpSolver(lbx=0., p=data_theta_batch.T)
        # print(self.qpSolver)
        # end_QP_time = time.time()
        # print("QP_time: " + str(end_QP_time - start_QP_time))
        loss_batch = sol['f']
        lam_phi_batch = sol['x']
        lam_batch = lam_phi_batch[0:self.n_lam, :]
        phi_batch = lam_phi_batch[self.n_lam:, :]
        mu_batch = sol['lam_x']
        # for i in range(batch_size):
        #     F_learnt = self.F_fn(current_theta)
        #     F_state = batch_F[i]
        #     F_check = F_learnt + F_state
        #     # print(F_learnt)
        #     # print(F_state)
        #     print(min(np.linalg.eigvals(F_check+F_check.T)))
        #     print(min(np.linalg.eigvals(F_state + F_state.T)))
        # pdb.set_trace()

        # solve the gradient
        start_gradient_time = time.time()
        dtheta_batch, dyn_loss_batch, lcp_loss_batch, = self.loss_fn(data_batch.T, lam_phi_batch, mu_batch,
                                                                     current_theta, theta_M_batch.T)
        end_gradient_time = time.time()
        print("Gradient_time: " + str(end_gradient_time - start_gradient_time))

        # do the update of gradient descent
        mean_loss = loss_batch
        mean_dtheta = dtheta_batch.full().mean(axis=1)
        mean_dyn_loss = dyn_loss_batch
        mean_lcp_loss = lcp_loss_batch

        end_step_time = time.time()
        print("Step_time: " + str(end_step_time - start_step_time))
        return mean_loss, mean_dtheta, mean_dyn_loss, mean_lcp_loss, lam_batch.T
