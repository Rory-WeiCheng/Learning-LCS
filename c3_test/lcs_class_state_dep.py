import pdb
import time
import casadi
import numpy as np
from casadi import *
from scipy import interpolate
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

class LCS_Gen:

    def __init__(self, n_state, n_control, n_lam, A=None, B=None, C=None, dyn_offset=None,
                 D=None, E=None, F=None, lcp_offset=None):
        self.n_state = n_state
        self.n_control = n_control
        self.n_lam = n_lam

        self.A = DM(A)
        self.B = DM(B)
        self.C = DM(C)
        self.dyn_offset = DM(dyn_offset)
        self.D = DM(D)
        self.E = DM(E)
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
        Model_para+= [vec(A_M)]
        B_M = SX.sym('B_M', self.n_state, self.n_control)
        Model_para += [vec(B_M)]
        C_M = SX.sym('C_M', self.n_state, self.n_lam)
        Model_para += [vec(C_M)]
        dyn_offset_M = SX.sym('dyn_offset_M', self.n_state)
        Model_para += [vec(dyn_offset_M)]
        D_M = SX.sym('D_M', self.n_lam, self.n_state)
        Model_para += [vec(D_M)]
        E_M = SX.sym('E_M', self.n_lam, self.n_control)
        Model_para += [vec(E_M)]
        F_M = SX.sym('F_M', self.n_lam, self.n_lam)
        Model_para += [vec(F_M)]
        lcp_offset_M = SX.sym('lcp_offset_M', self.n_lam)
        Model_para += [vec(lcp_offset_M)]
        theta_M = vcat(Model_para)
        xu_theta_pair = vertcat(xu_pair, theta_M)

        # dynamics
        dyn = (self.A+A_M) @ x + (self.B+B_M) @ u + (self.C+C_M) @ lam + (self.dyn_offset+dyn_offset_M)
        self.dyn_fn = Function('dyn_fn', [xu_theta_pair, lam], [dyn])

        # lcp function
        lcp_cstr = (self.D+D_M) @ x + (self.E+E_M) @ u + (self.F+F_M) @ lam + (self.lcp_offset+lcp_offset_M)
        lcp_loss = dot(lam, lcp_cstr)
        self.lcp_loss_fn = Function('dyn_loss_fn', [xu_theta_pair, lam], [lcp_loss])
        self.lcp_cstr_fn = Function('dis_cstr_fn', [xu_theta_pair, lam], [lcp_cstr])

        # establish the qp solver to solve for lcp
        xu_theta = vertcat(xu_theta_pair)
        quadprog = {'x': lam, 'f': lcp_loss, 'g': lcp_cstr, 'p': xu_theta}
        opts = {'print_time': 0, 'osqp': {'verbose': False}}
        self.lcpSolver = qpsol('lcpSolver', 'osqp', quadprog, opts)

    def nextState(self, x_batch, u_batch, batch_A, batch_B, batch_C, batch_dynamic_offset,
             batch_D, batch_E, batch_F, batch_lcp_offset):

        batch_size = x_batch.shape[0]
        theta_M_list = []
        for i in range(batch_size):
            A_Mi = batch_A[i].flatten('F')
            B_Mi = batch_B[i].flatten('F')
            C_Mi = batch_C[i].flatten('F')
            dynamic_offset_Mi = batch_dynamic_offset[i]
            D_Mi = batch_D[i].flatten('F')
            E_Mi = batch_E[i].flatten('F')
            F_Mi = batch_F[i].flatten('F')
            lcp_offset_Mi = batch_lcp_offset[i]
            theta_Mi = np.concatenate([A_Mi, B_Mi, C_Mi, dynamic_offset_Mi, D_Mi, E_Mi, F_Mi, lcp_offset_Mi])
            theta_M_list.append(theta_Mi)
        theta_M_batch = np.stack(theta_M_list,axis=1).T
        xu_theta_batch = np.hstack((x_batch, u_batch, theta_M_batch))

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
class LCS_VN:

    def __init__(self, n_state, n_control, n_lam, A=None, B=None, C=None, dyn_offset=None,
                 D=None, E=None, G_para=None, H=None, lcp_offset=None, F_stiffness=1.0):
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

        if C is None:
            self.C = SX.sym('C', n_state, n_lam)
            self.tunable_para += [vec(self.C)]
        else:
            self.C = DM(C)

        if dyn_offset is None:
            self.dyn_offset = SX.sym('dyn_offset', n_state)
            self.tunable_para += [vec(self.dyn_offset)]
        else:
            self.dyn_offset = DM(dyn_offset)

        if D is not None:
            self.D = DM(D)
        else:
            self.D = SX.sym('D', n_lam, n_state)
            self.tunable_para += [vec(self.D)]

        if E is None:
            self.E = SX.sym('E', n_lam, n_control)
            self.tunable_para += [vec(self.E)]
        else:
            self.E = DM(E)

        if G_para is not None:
            self.G_para = DM(G_para)
            self.G = self.toMatG(G_para)
        else:
            self.G_para = SX.sym('G_para', int((self.n_lam + 1) * self.n_lam / 2))
            self.G = self.toMatG(self.G_para)
            self.tunable_para += [vec(self.G_para)]

        if H is None:
            self.H = SX.sym('H', n_lam, n_lam)
            self.tunable_para += [vec(self.H)]
        else:
            self.H = DM(H)

        if lcp_offset is None:
            self.lcp_offset = SX.sym('lcp_offset', n_lam)
            self.tunable_para += [vec(self.lcp_offset)]
        else:
            self.lcp_offset = DM(lcp_offset)

        self.theta = vcat(self.tunable_para)
        self.n_theta = self.theta.numel()

        self.F = self.G @ self.G.T + F_stiffness * np.eye(self.n_lam) + self.H - self.H.T
        self.F_fn = Function('F_fn', [self.theta], [self.F])
        self.E_fn = Function('E_fn', [self.theta], [self.E])
        self.G_fn = Function('F_fn', [self.theta], [self.G])
        self.A_fn = Function('A_fn', [self.theta], [self.A])
        self.B_fn = Function('B_fn', [self.theta], [self.B])
        self.C_fn = Function('C_fn', [self.theta], [self.C])
        self.D_fn = Function('D_fn', [self.theta], [self.D])
        self.dyn_offset_fn = Function('dyn_offset_fn', [self.theta], [self.dyn_offset])
        self.lcp_offset_fn = Function('lcp_offset_fn', [self.theta], [self.lcp_offset])

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

    def diff(self, gamma=1e-2, epsilon=1., F_ref=0., w_F=0., C_ref=0., w_C=0.):

        # define the system variable
        x = SX.sym('x', self.n_state)
        u = SX.sym('u', self.n_control)
        x_next = SX.sym('x_next', self.n_state)

        # define the model system matrices, M stands for model
        Model_para = []
        A_M = SX.sym('A_M', self.n_state, self.n_state)
        Model_para+= [vec(A_M)]
        B_M = SX.sym('B_M', self.n_state, self.n_control)
        Model_para += [vec(B_M)]
        C_M = SX.sym('C_M', self.n_state, self.n_lam)
        Model_para += [vec(C_M)]
        dyn_offset_M = SX.sym('dyn_offset_M', self.n_state)
        Model_para += [vec(dyn_offset_M)]
        D_M = SX.sym('D_M', self.n_lam, self.n_state)
        Model_para += [vec(D_M)]
        E_M = SX.sym('E_M', self.n_lam, self.n_control)
        Model_para += [vec(E_M)]
        F_M = SX.sym('F_M', self.n_lam, self.n_lam)
        Model_para += [vec(F_M)]
        lcp_offset_M = SX.sym('lcp_offset_M', self.n_lam)
        Model_para += [vec(lcp_offset_M)]

        # vectorize the matrices to be theta_M
        theta_M = vcat(Model_para)
        n_theta_M = theta_M.numel()

        data_pair = vertcat(x, u, x_next)

        # define the variables for LCP
        lam = SX.sym('lam', self.n_lam)
        phi = SX.sym('phi', self.n_lam)

        # dynamics
        dyn = (self.A + A_M) @ x + (self.B + B_M) @ u + (self.C + C_M) @ lam + (self.dyn_offset + dyn_offset_M)
        dyn_loss = dot(dyn - x_next, dyn - x_next)

        # lcp loss
        dist = (self.D + D_M) @ x + (self.E + E_M) @ u + self.F @ lam + self.lcp_offset
        lcp_aug_loss = dot(lam, phi) + 1 / gamma * dot(phi - dist, phi - dist)

        # define loss function
        total_loss = 1. * dyn_loss \
                     + 1. / epsilon * lcp_aug_loss \
                     + w_C * dot(vec(DM(C_ref)) - vec(self.C), vec(DM(C_ref)) - vec(self.C)) \
                     + w_F * dot(vec(DM(F_ref)) - vec(self.F), vec(DM(F_ref)) - vec(self.F))

        # establish the qp solver, now the data_theta should contain three parts:
        # 1. (x,u,x_next) 2. theta (learnable parameters) 3.theta_M (local state dependent lcs matrices input)
        data_theta = vertcat(data_pair, self.theta, theta_M)
        lam_phi = vertcat(lam, phi)
        quadprog = {'x': lam_phi, 'f': total_loss, 'p': data_theta}
        opts = {'print_time': 0, 'osqp': {'verbose': False}}
        self.qpSolver = qpsol('Solver_QP', 'osqp', quadprog, opts)

        # compute the jacobian from lam to theta
        mu = SX.sym('mu', self.n_lam + self.n_lam)
        L = total_loss + dot(mu, lam_phi)

        # compute the derivative of the value function with respect to theta using method 2
        dyn_loss_plus = dyn_loss + \
                        w_C * dot(vec(DM(C_ref)) - vec(self.C), vec(DM(C_ref)) - vec(self.C)) + \
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

    def step(self, batch_x, batch_u, batch_x_next, current_theta, batch_A, batch_B, batch_C, batch_dynamic_offset,
             batch_D, batch_E, batch_F, batch_lcp_offset):

        # do data processing for batch calculation
        # batch x,u,x_next, horizontal concatenate, then use transpose for input
        batch_size = batch_x.shape[0]
        data_batch = np.hstack((batch_x, batch_u, batch_x_next))
        # theta and theta_M
        theta_batch = np.tile(current_theta, (batch_size, 1))
        theta_M_list = []
        for i in range(batch_size):
            A_Mi = batch_A[i].flatten('F')
            B_Mi = batch_B[i].flatten('F')
            C_Mi = batch_C[i].flatten('F')
            dynamic_offset_Mi = batch_dynamic_offset[i]
            D_Mi = batch_D[i].flatten('F')
            E_Mi = batch_E[i].flatten('F')
            F_Mi = batch_F[i].flatten('F')
            lcp_offset_Mi = batch_lcp_offset[i]
            theta_Mi = np.concatenate([A_Mi, B_Mi, C_Mi, dynamic_offset_Mi, D_Mi, E_Mi, F_Mi, lcp_offset_Mi])
            theta_M_list.append(theta_Mi)
        theta_M_batch = np.stack(theta_M_list,axis=1).T

        # Assemble data (x,u,x_next) and theta (learnt + data input)
        data_theta_batch = np.hstack((batch_x, batch_u, batch_x_next, theta_batch, theta_M_batch))

        # establish the solver, solve for lambda and phi using QP and prepare to feed to the gradient step
        sol = self.qpSolver(lbx=0., p=data_theta_batch.T)
        loss_batch = sol['f'].full()
        lam_phi_batch = sol['x'].full()
        lam_batch = lam_phi_batch[0:self.n_lam, :]
        phi_batch = lam_phi_batch[self.n_lam:, :]
        mu_batch = sol['lam_x'].full()

        # solve the gradient
        dtheta_batch, dyn_loss_batch, lcp_loss_batch, = self.loss_fn(data_batch.T, lam_phi_batch, mu_batch,
                                                                     current_theta,theta_M_batch.T)

        # do the update of gradient descent
        mean_loss = loss_batch.mean()
        mean_dtheta = dtheta_batch.full().mean(axis=1)
        mean_dyn_loss = dyn_loss_batch.full().mean()
        mean_lcp_loss = lcp_loss_batch.full().mean()

        return mean_loss, mean_dtheta, mean_dyn_loss, mean_lcp_loss, lam_batch.T

    def pred(self, batch_x, batch_u, batch_x_next, current_theta):
        # solve for the
        batch_size = batch_x.shape[0]
        theta_batch = np.tile(current_theta, (batch_size, 1))
        xu_theta_batch = np.hstack((batch_x, batch_u, theta_batch))

        sol = self.pred_qpSolver(lbg=0., p=xu_theta_batch.T)
        x_next_lam_phi_batch = sol['x'].full()
        x_next_batch = x_next_lam_phi_batch[0:self.n_state, :]
        lam_batch = x_next_lam_phi_batch[self.n_state:self.n_state + self.n_lam, :]
        phi_batch = x_next_lam_phi_batch[self.n_state + self.n_lam:, :]

        # prediction error
        pred_error_batch = self.pred_error_fn(batch_x_next.T, x_next_batch).full().flatten()

        return pred_error_batch.mean()
