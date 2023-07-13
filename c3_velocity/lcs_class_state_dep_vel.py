import pdb
import time
import casadi
import numpy as np
from casadi import *
from scipy import interpolate
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import scipy

# 2023.4.24 fix the notation issue, follow Alp's work
class LCS_Gen:

    def __init__(self, n_state, n_control, n_lam, n_vel, A=None, B=None, D=None, dyn_offset=None,
                 E=None, H=None, F=None, lcp_offset=None):
        self.n_state = n_state
        self.n_control = n_control
        self.n_lam = n_lam
        self.n_vel = n_vel

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
        x_next = SX.sym('x_next', self.n_vel)

        # define the model system matrices, M stands for model
        Model_para = []
        A_M = SX.sym('A_M', self.n_vel, self.n_state)
        Model_para+= [vec(A_M)]
        B_M = SX.sym('B_M', self.n_vel, self.n_control)
        Model_para += [vec(B_M)]
        D_M = SX.sym('D_M', self.n_vel, self.n_lam)
        Model_para += [vec(D_M)]
        dyn_offset_M = SX.sym('dyn_offset_M', self.n_vel)
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
        dyn = (self.A+A_M) @ x + (self.B+B_M) @ u + (self.D+D_M) @ lam + (self.dyn_offset+dyn_offset_M)
        self.dyn_fn = Function('dyn_fn', [xu_theta_pair, lam], [dyn])

        # lcp function
        lcp_cstr = (self.E+E_M) @ x + (self.H+H_M) @ u + (self.F+F_M) @ lam + (self.lcp_offset+lcp_offset_M)
        lcp_loss = dot(lam, lcp_cstr)
        self.lcp_loss_fn = Function('dyn_loss_fn', [xu_theta_pair, lam], [lcp_loss])
        self.lcp_cstr_fn = Function('dis_cstr_fn', [xu_theta_pair, lam], [lcp_cstr])

        # establish the qp solver to solve for lcp
        xu_theta = vertcat(xu_theta_pair)
        quadprog = {'x': lam, 'f': lcp_loss, 'g': lcp_cstr, 'p': xu_theta}
        # use QPOASES to get strict feasible solution
        opts = {"print_time": 0, "printLevel": "none", "verbose": 0, 'error_on_fail': False}
        self.qpSolver = qpsol('Solver_QP', 'qpoases', quadprog, opts)

    def nextState(self, x_batch, u_batch, batch_A, batch_B, batch_D, batch_dynamic_offset,
             batch_E, batch_H, batch_F, batch_lcp_offset):

        batch_size = x_batch.shape[0]

        theta_M_batch = self.Form_theta_M(batch_A, batch_B, batch_D, batch_dynamic_offset, batch_E, batch_H, batch_F,
                                          batch_lcp_offset).full().T
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
# 2023.4.24 fix the notation issue, follow Alp's work
class LCS_VN:

    def __init__(self, n_state, n_control, n_lam, n_vel, A=None, B=None, D=None, dyn_offset=None,
                 E=None, H=None, G_para=None, S=None, lcp_offset=None, F_stiffness=1.0, Q=None, A_mask=None,
                 B_mask=None, D_mask=None, dyn_offset_mask=None, E_mask=None, H_mask=None, G_mask=None,
                 S_mask=None, lcp_offset_mask=None):
        # system parameters
        self.n_state = n_state
        self.n_control = n_control
        self.n_lam = n_lam
        self.n_vel = n_vel

        # weighting matrix for prediction loss
        self.Q = Q

        # define system matrices
        self.tunable_para = []
        if A is None:
            self.A = SX.sym('A', n_vel, n_state)
            self.tunable_para += [vec(self.A)]
        else:
            self.A = DM(A)

        if B is None:
            self.B = SX.sym('B', n_vel, n_control)
            self.tunable_para += [vec(self.B)]
        else:
            self.B = DM(B)

        if D is None:
            self.D = SX.sym('D', n_vel, n_lam)
            self.tunable_para += [vec(self.D)]
        else:
            self.D = DM(D)

        if dyn_offset is None:
            self.dyn_offset = SX.sym('dyn_offset', n_vel)
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

        self.F = self.G @ self.G.T + F_stiffness * SX.eye(self.n_lam) + self.S - self.S.T

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

        # define the mask matrices to impose the structure on the learnt matrices
        # notice the order should be consistent with the parameter definition above
        self.mask_para = []
        if A_mask is not None:
            self.A_mask = DM(A_mask)
            self.mask_para += [vec(self.A_mask)]
        if B_mask is not None:
            self.B_mask = DM(B_mask)
            self.mask_para += [vec(self.B_mask)]
        if D_mask is not None:
            self.D_mask = DM(D_mask)
            self.mask_para += [vec(self.D_mask)]
        if dyn_offset_mask is not None:
            self.dyn_offset_mask = DM(dyn_offset_mask)
            self.mask_para += [vec(self.dyn_offset_mask)]

        if E_mask is not None:
            self.E_mask = DM(E_mask)
            self.mask_para += [vec(self.E_mask)]
        if H_mask is not None:
            self.H_mask = DM(H_mask)
            self.mask_para += [vec(self.H_mask)]
        if G_mask is not None:
            self.G_mask = DM(G_mask)
            self.mask_para += [vec(self.G_mask)]
        if S_mask is not None:
            self.S_mask = DM(S_mask)
            self.mask_para += [vec(self.S_mask)]
        if lcp_offset_mask is not None:
            self.lcp_offset_mask = DM(lcp_offset_mask)
            self.mask_para += [vec(self.lcp_offset_mask)]

        self.theta_mask = vcat(self.mask_para)
        self.n_theta_mask = self.theta_mask.numel()
        if not (self.n_theta_mask == self.n_theta):
            raise Exception('check the parameter and mask dimension consistency')

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
        x_next = SX.sym('x_next', self.n_vel)

        # define the model system matrices, M stands for model
        Model_para = []
        A_M = SX.sym('A_M', self.n_vel, self.n_state)
        Model_para+= [vec(A_M)]
        B_M = SX.sym('B_M', self.n_vel, self.n_control)
        Model_para += [vec(B_M)]
        D_M = SX.sym('D_M', self.n_vel, self.n_lam)
        Model_para += [vec(D_M)]
        dyn_offset_M = SX.sym('dyn_offset_M', self.n_vel)
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
        self.Form_theta_M = Function('Form_theta_M', [A_M, B_M, D_M, dyn_offset_M, E_M, H_M, F_M, lcp_offset_M], [theta_M])
        n_theta_M = theta_M.numel()

        data_pair = vertcat(x, u, x_next)

        # define the variables for LCP
        lam = SX.sym('lam', self.n_lam)
        phi = SX.sym('phi', self.n_lam)

        # dynamics
        dyn = (self.A + A_M) @ x + (self.B + B_M) @ u + (self.D + D_M) @ lam + (self.dyn_offset + dyn_offset_M)
        dyn_error = dyn - x_next
        if self.Q is None:
            dyn_loss = 0.5 * dot(dyn - x_next, dyn - x_next)
        else:
            dyn_loss = 0.5 * (dyn - x_next).T @ self.Q @ (dyn - x_next)

        # lcp loss
        dist = (self.E + E_M) @ x + (self.H + H_M) @ u + (self.F + F_M) @ lam + (self.lcp_offset + lcp_offset_M)
        lcp_error = dist - phi
        lcp_aug_loss = dot(lam, phi) + 0.5 * 1 / gamma * dot(phi - dist, phi - dist)

        # define loss function
        total_loss = 1. * dyn_loss \
                     + (1. / epsilon) * lcp_aug_loss \
                     # + w_D * dot(vec(DM(D_ref)) - vec(self.D), vec(DM(D_ref)) - vec(self.D)) \
                     # + w_F * dot(vec(DM(F_ref)) - vec(self.F), vec(DM(F_ref)) - vec(self.F)) \

        # # construct the quadratic term for convexity check, self coded
        # D_QP = (self.D + D_M)
        # F_QP = (self.F + F_M)
        # Quad_blk_ul = D_QP.T @ self.Q @ D_QP + 1 / (epsilon * gamma) * F_QP.T @ F_QP
        # Quad_blk_ur = 1 / (epsilon) * SX.eye(self.n_lam) - 1 / (epsilon * gamma) * F_QP.T
        # Quad_blk_bl = 1 / (epsilon) * SX.eye(self.n_lam) - 1 / (epsilon * gamma) * F_QP
        # Quad_blk_br = 1 / (epsilon * gamma) * SX.eye(self.n_lam)
        # Quad = blockcat(Quad_blk_ul, Quad_blk_ur, Quad_blk_bl, Quad_blk_br)
        # self.Form_Quadratic = Function('Form_Quadratic', [self.theta, theta_M], [Quad])
        # self.Form_Fcheck = Function('Form_Fcheck', [F_M, self.G, self.S], [F_QP])

        # establish the qp solver, now the data_theta should contain three parts:
        # 1. (x,u,x_next) 2. theta (learnable parameters) 3.theta_M (local state dependent lcs matrices input)
        data_theta = vertcat(data_pair, self.theta, theta_M)
        lam_phi = vertcat(lam, phi)

        # construct the quadratic term for convexity check, using casadi
        Quadratic_casadi, self.Linear_casadi, self.constant_casadi = quadratic_coeff(total_loss, lam_phi)
        self.Form_Quadratic_casadi = Function('Form_Quadratic_casadi', [self.theta, theta_M], [Quadratic_casadi])

        quadprog = {'x': lam_phi, 'f': total_loss, 'p': data_theta}

        # # NLP (for Stwart-Trinkle)
        # nonlinprog = {'x': lam_phi, 'f': total_loss, 'p': data_theta}
        # opts = {'print_time': 0, 'ipopt': {'print_level': 0},'error_on_fail': False}
        # self.nlpSolver = nlpsol('Solver_NLP', 'ipopt', nonlinprog, opts)

        # # OSQP
        # opts = {'print_time': 0, 'osqp': {'verbose': False}, 'error_on_fail': False}
        # opts = {'print_time': 0, 'osqp': {'verbose': False, 'eps_rel': 1e-8},'error_on_fail':False}
        # self.qpSolver = qpsol('Solver_QP', 'osqp', quadprog, opts)

        # QPOASES
        opts = {"print_time": 0, "printLevel": "none", "verbose": 0, 'error_on_fail': False}
        self.qpSolver = qpsol('Solver_QP', 'qpoases', quadprog, opts)


        # compute the jacobian from lam to theta
        mu = SX.sym('mu', self.n_lam + self.n_lam)
        L = total_loss + dot(mu, lam_phi)

        # compute the derivative of the value function with respect to theta using method 2
        dyn_loss_plus = dyn_loss + \
                        w_D * dot(vec(DM(D_ref)) - vec(self.D), vec(DM(D_ref)) - vec(self.D)) + \
                        w_F * dot(vec(DM(F_ref)) - vec(self.F), vec(DM(F_ref)) - vec(self.F))
        Dlambda = (self.D + D_M) @ lam
        self.loss_fn = Function('loss_fn', [data_pair, lam_phi, mu, self.theta, theta_M],
                                [jacobian(L, self.theta).T*self.theta_mask, dyn_loss_plus, lcp_aug_loss, dyn_error, lcp_error, Dlambda])
        # pdb.set_trace()

    def step(self, batch_x, batch_u, batch_x_next, current_theta, batch_A, batch_B, batch_D, batch_dynamic_offset,
             batch_E, batch_H, batch_F, batch_lcp_offset):

        # do data processing for batch calculation
        # batch x,u,x_next, horizontal concatenate, then use transpose for input
        batch_size = batch_x.shape[0]
        data_batch = np.hstack((batch_x, batch_u, batch_x_next))
        # theta and theta_M
        theta_batch = np.tile(current_theta, (batch_size, 1))

        theta_M_batch = self.Form_theta_M(batch_A, batch_B, batch_D, batch_dynamic_offset, batch_E, batch_H,
                                          batch_F, batch_lcp_offset).full().T
        # pdb.set_trace()
        # Assemble data (x,u,x_next) and theta (learnt + data input)
        data_theta_batch = np.hstack((batch_x, batch_u, batch_x_next, theta_batch, theta_M_batch))

        # establish the solver, solve for lambda and phi using QP and prepare to feed to the gradient step
        sol = self.qpSolver(lbx=0, p=data_theta_batch.T)
        # sol = self.nlpSolver(lbx=0, p=data_theta_batch.T)

        loss_batch = sol['f'].full()
        lam_phi_batch = sol['x'].full()
        lam_batch = lam_phi_batch[0:self.n_lam, :]
        phi_batch = lam_phi_batch[self.n_lam:, :]
        mu_batch = sol['lam_x'].full()

        # set the parameter data type to DM to speed up the code
        data_batch = DM(data_batch)
        theta_M_batch = DM(theta_M_batch)
        lam_phi_batch = DM(lam_phi_batch)
        mu_batch = DM(mu_batch)

        # solve the gradient
        # start_gradient_time = time.time()
        dtheta_batch, dyn_loss_batch, lcp_loss_batch, dyn_error_batch, lcp_error_batch, Dlambda = \
            self.loss_fn(data_batch.T, lam_phi_batch, mu_batch, current_theta, theta_M_batch.T)

        # calculate the average of the batch gradient for the update of gradient descent
        # mean_dtheta = dtheta_batch.full().mean(axis=1)
        mean_loss = loss_batch.mean()
        mean_dtheta = dtheta_batch.full()

        # gradient magnitude filtering, need to analyze the distribution of gradient
        mean_dtheta = np.where(abs(mean_dtheta)>1e-10, mean_dtheta, 0)
        mean_dtheta = mean_dtheta.mean(axis=1)

        mean_dyn_loss = dyn_loss_batch.full().mean()
        mean_lcp_loss = lcp_loss_batch.full().mean()

        Dlambda_mean = Dlambda.full().mean(axis=1)

        return mean_loss, mean_dtheta, mean_dyn_loss, mean_lcp_loss, lam_batch.T, dyn_error_batch.T, lcp_error_batch.T, Dlambda_mean
