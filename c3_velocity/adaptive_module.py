import sys

# hack the sys path to let it be able to be called from procman
# print the sys path when using pycharm and record the sys path as below
# print (sys.path)
sys.path = ['/usr/rory-workspace/Learning-LCS/c3_velocity', '/usr/rory-workspace/Learning-LCS', '/usr/lib/python38.zip', '/usr/lib/python3.8', '/usr/lib/python3.8/lib-dynload', '/usr/rory-workspace/Learning-LCS/venv/lib/python3.8/site-packages', '/usr/rory-workspace/Learning-LCS']

import lcs_class_state_dep_vel
import lcs.optim as opt
# from lcs import optim as opt
import numpy as np
import scipy
import time
import pdb
import lcm
from lcm_types.lcm_adaptive import lcmt_lcs, lcmt_robot_output, lcmt_c3, lcmt_learning_data
import threading
import matplotlib.pyplot as plt

############################################## data setting #########################################################
num_state = 19
num_velocity = 9
num_control = 3
num_lambda = 12

# initialization of the residual lcs, all to be zeros and can be first published
A_res = np.zeros((num_velocity, num_state))
B_res = np.zeros((num_velocity, num_control))
D_res = np.zeros((num_velocity, num_lambda))
d_res = np.zeros(num_velocity)

E_res = np.zeros((num_lambda, num_state))
F_res = np.zeros((num_lambda, num_lambda))
H_res = np.zeros((num_lambda, num_control))
c_res = np.zeros(num_lambda)

############################################## LCM and data buffer setting #############################################
class LCM_data_buffer:
    def __init__(self):
        self.lc = lcm.LCM()

        self.A_list = []
        self.B_list = []
        self.D_list = []
        self.d_list = []

        self.E_list = []
        self.F_list = []
        self.H_list = []
        self.c_list = []

        self.input_list = []
        self.state_list = []
        self.state_pred_list = []

        self.timestamp_list = []
        self.settling_time = 0

    def learning_data_receiver(self, channel, data):
        # decode message
        msg = lcmt_learning_data.decode(data)

        if msg.utime * 1e-6 < msg.settling_time:
            return
        else:
            # grab data and store in the list
            self.A_list.append(np.array(msg.A))
            self.B_list.append(np.array(msg.B))
            self.D_list.append(np.array(msg.D))
            self.d_list.append(np.array(msg.d))

            self.E_list.append(np.array(msg.E))
            self.F_list.append(np.array(msg.F))
            self.H_list.append(np.array(msg.H))
            self.c_list.append(np.array(msg.c))

            self.state_list.append(np.array(msg.state))
            self.state_pred_list.append(np.array(msg.state_pred))
            self.input_list.append(np.array(msg.input))

            self.timestamp_list.append(msg.utime * 1e-6)
            self.settling_time = msg.settling_time


    # def lcs_receiver(self, channel, data):
    #     msg = lcmt_lcs.decode(data)
    #     self.A_list.append = np.array(msg.A)
    #
    # def c3_input_receiver(self,channel, data):
    #     msg = lcmt_c3.decode(data)
    #     c3_input = np.array(msg.data[-3:])
    #     self.input_list.append(c3_input)
    #
    # def robot_state_receiver(self,channel, data):
    #     msg = lcmt_robot_output.decode(data)
    #     position = np.array(msg.position)
    #     self.state_list.append(position)

lcm_data = LCM_data_buffer()
subscription_robot_output = lcm_data.lc.subscribe("LEARNING_DATASET", lcm_data.learning_data_receiver)
lc_publish = lcm.LCM()

############################################## Learner setting #########################################################
training_data_size = 100
cnt = 0

# warm start options
A_init = np.zeros(num_velocity * num_state)
B_init = np.zeros(num_velocity * num_control)
D_init = np.zeros(num_velocity * num_lambda)
d_init = np.zeros(num_velocity)

E_init = np.zeros(num_lambda * num_state)
F_init = np.zeros(num_lambda * num_lambda)
H_init = np.zeros(num_lambda * num_control)
c_init = np.zeros(num_lambda)
# reparameterization part F divided into G and H_re
G_init = np.zeros(int((num_lambda + 1) * num_lambda / 2))
S_init = np.zeros(num_lambda * num_lambda)
vn_curr_theta = np.concatenate([A_init,B_init,D_init,d_init,E_init,H_init,G_init,S_init,c_init])
# pdb.set_trace()

# # mask option, enforcing the learning structure, the masked entry (0) would remain the initial value
# A_mask = np.ones((num_velocity, num_state))
# B_mask = np.ones((num_velocity, num_control))
# D_mask = np.ones((num_velocity, num_lambda))
# D_mask[:,0:2] = np.zeros((num_velocity, 2))
# d_mask = np.ones(num_velocity)
#
# E_mask = np.ones((num_lambda, num_state))
# E_mask[0:2,:] = np.zeros((2, num_state))
# H_mask = np.ones((num_lambda, num_control))
# H_mask[0:2,:] = np.zeros((2, num_control))
# c_mask = np.ones(num_lambda)
# c_mask[0:2] = 0
#
# # need to think carefully later about this part
# G_mask = np.ones(int((num_lambda + 1) * num_lambda / 2))
# S_mask = np.ones((num_lambda,num_lambda))

# # only learn c[2] and c[3]
A_mask = np.zeros((num_velocity, num_state))
B_mask = np.zeros((num_velocity, num_control))
D_mask = np.zeros((num_velocity, num_lambda))
d_mask = np.zeros(num_velocity)

E_mask = np.zeros((num_lambda, num_state))
H_mask = np.zeros((num_lambda, num_control))
c_mask = np.zeros(num_lambda)
c_mask[2] = 1
c_mask[3] = 1

G_mask = np.zeros(int((num_lambda + 1) * num_lambda / 2))
S_mask = np.zeros((num_lambda,num_lambda))

# establish the violation-based learner, assign weight for each residual
F_stiffness = 0.0
gamma = 1e-1
epsilon = 1e0
Q_ee = 0 * np.eye(3)
Q_br = 0 * np.eye(3)
# Q_bp = np.eye(3)
Q_bp = np.diag(np.array([1,1,1]))
Q = scipy.linalg.block_diag(Q_ee,Q_br,Q_bp)

# pdb.set_trace()
vn_learner = lcs_class_state_dep_vel.LCS_VN(n_state=num_state, n_control=num_control, n_lam=num_lambda, n_vel=num_velocity,
                                            F_stiffness=F_stiffness, Q=Q, A_mask=A_mask, B_mask=B_mask, D_mask=D_mask,
                                            dyn_offset_mask=d_mask, E_mask=E_mask, H_mask=H_mask, G_mask=G_mask, S_mask=S_mask,
                                            lcp_offset_mask=c_mask)
vn_learner.diff(gamma=gamma, epsilon=epsilon, w_D=1e-6, D_ref=0, w_F=0e-6, F_ref=0)

# establish the optimizer
# vn_learning_rate = 0.1e-5
vn_learning_rate = 0.1e-4
vn_optimizier = opt.Adam()
vn_optimizier.learning_rate = vn_learning_rate

# training loop
max_iter = 10
mini_batch_size = 10

# storing loss
total_loss_list = []
dyn_loss_list = []
lcp_loss_list = []
theta_learnt_list = []

########################################## multiple processing #########################################################
def data_grabbing():
    global lcm_data
    global cnt
    while True:
        lcm_data.lc.handle()
        # if cnt == 100:
        #     pdb.set_trace()

def learning():
    global cnt
    global num_state,num_velocity,num_control,num_lambda
    global A_res,B_res,D_res,d_res,E_res,F_res,H_res,c_res
    global max_iter, mini_batch_size, Q
    global vn_learner, vn_optimizier, vn_curr_theta
    global lcm_data, lc_publish
    global total_loss_list, dyn_loss_list, lcp_loss_list, theta_learnt_list

    while True:
        ## publishing test
        msg = lcmt_lcs()
        msg.num_state = num_state
        msg.num_velocity = num_velocity
        msg.num_control = num_control
        msg.num_lambda = num_lambda

        msg.A = A_res
        msg.B = B_res
        msg.D = D_res
        msg.d = d_res

        msg.E = E_res
        msg.F = F_res
        msg.H = H_res
        msg.c = c_res

        lc_publish.publish("RESIDUAL_LCS", msg.encode())
        if len(lcm_data.state_list) < (cnt+1)*mini_batch_size + 1:
            continue
        else:
            start_time = time.time()
            x_batch = np.array(lcm_data.state_list[cnt*mini_batch_size: (cnt+1)*mini_batch_size])
            u_batch = np.array(lcm_data.input_list[cnt*mini_batch_size: (cnt+1)*mini_batch_size])
            x_pred_batch = np.array(lcm_data.state_pred_list[cnt*mini_batch_size: (cnt+1)*mini_batch_size])
            x_next_batch = np.array(lcm_data.state_list[cnt*mini_batch_size + 1: (cnt+1)*mini_batch_size + 1])
            res_batch = x_next_batch[:,num_state-num_velocity:] - x_pred_batch
            # if cnt > 100:
            #     res_batch[:,8]= np.zeros(mini_batch_size)
            # pdb.set_trace()
            # pdb.set_trace()
            # print(cnt*mini_batch_size + 1)
            # print((cnt+1)*mini_batch_size + 1)

            A_batch = np.zeros((num_velocity, mini_batch_size * num_state))
            B_batch = np.zeros((num_velocity, mini_batch_size * num_control))
            D_batch = np.array(lcm_data.D_list[cnt*mini_batch_size: (cnt+1)*mini_batch_size]).swapaxes(0, 1).reshape(num_velocity, -1)
            d_batch = np.zeros((num_velocity, mini_batch_size))

            E_batch = np.array(lcm_data.E_list[cnt*mini_batch_size: (cnt+1)*mini_batch_size]).swapaxes(0, 1).reshape(num_lambda, -1)
            F_batch = np.array(lcm_data.F_list[cnt*mini_batch_size: (cnt+1)*mini_batch_size]).swapaxes(0, 1).reshape(num_lambda, -1)
            H_batch = np.array(lcm_data.H_list[cnt*mini_batch_size: (cnt+1)*mini_batch_size]).swapaxes(0, 1).reshape(num_lambda, -1)
            c_batch = np.array(lcm_data.c_list[cnt*mini_batch_size: (cnt+1)*mini_batch_size]).T

            end_data_time = time.time()

            for iter in range(max_iter):
                vn_mean_loss, vn_dtheta, vn_dyn_loss, vn_lcp_loss, _ = vn_learner.step(batch_x=x_batch, batch_u=u_batch,
                    batch_x_next=res_batch, current_theta=vn_curr_theta, batch_A=A_batch, batch_B=B_batch, batch_D=D_batch,
                    batch_dynamic_offset=d_batch, batch_E=E_batch, batch_H=H_batch, batch_F=F_batch, batch_lcp_offset=c_batch)
                vn_curr_theta = vn_optimizier.step(vn_curr_theta, vn_dtheta)
                # pdb.set_trace()

            theta_learnt_list.append(vn_curr_theta)
            total_loss_list.append(vn_mean_loss)
            dyn_loss_list.append(vn_dyn_loss)
            lcp_loss_list.append(vn_lcp_loss)

            # print('iter:', iter, 'vn_loss: ', vn_mean_loss, 'vn_dyn_loss: ', vn_dyn_loss, 'vn_lcp_loss', vn_lcp_loss)
            # pdb.set_trace()

            Start_LCSdata_time = time.time()

            A_res = vn_learner.A_fn(vn_curr_theta).full()
            B_res = vn_learner.B_fn(vn_curr_theta).full()
            D_res = vn_learner.D_fn(vn_curr_theta).full()
            d_res = vn_learner.dyn_offset_fn(vn_curr_theta).full()
            E_res = vn_learner.E_fn(vn_curr_theta).full()
            F_res = vn_learner.F_fn(vn_curr_theta).full()
            H_res = vn_learner.H_fn(vn_curr_theta).full()
            c_res = vn_learner.lcp_offset_fn(vn_curr_theta).full()

            end_time = time.time()
            cnt = cnt + 1
        #     # print(cnt)
        #     # print("Learning Time：", end_time - start_time)
        #     # print("Data Time：", end_data_time - start_time + end_time - Start_LCSdata_time)
        #     # print(cnt*mini_batch_size)
        #     # print((cnt+1)*mini_batch_size)

def visualization():
    global cnt
    global mini_batch_size
    global lcm_data
    global total_loss_list, dyn_loss_list, lcp_loss_list, theta_learnt_list
    plt.figure(figsize=(24, 16))
    plt.ion()
    while True:
        if len(lcm_data.state_list) > 3:
            plt.clf()
            plot_length = len(lcm_data.state_list)-1
            x_pred_all = np.array(lcm_data.state_pred_list[0:plot_length])
            x_next_all = np.array(lcm_data.state_list[1:plot_length+1])
            # pdb.set_trace()
            res_all = x_next_all[:,num_state-num_velocity:] - x_pred_all
            # pdb.set_trace()
            plt.subplot(3, 1, 1)
            plt.plot(res_all[:,6],label='ball vx res')
            plt.legend(fontsize=20)
            plt.subplot(3, 1, 2)
            plt.plot(res_all[:,7],label='ball vy res')
            plt.legend(fontsize=20)
            plt.subplot(3, 1, 3)
            plt.plot(res_all[:,8],label='ball vz res')
            plt.legend(fontsize=20)
            plt.draw()
            plt.pause(0.01)
            # plt.plot(x_next_all[:,18],label='ball vz actual')
            # plt.legend(fontsize=20)
            # plt.plot(x_pred_all[:,8],label='ball vz pred')
            # plt.legend(fontsize=20)
            # plt.show()
            # print(cnt)


data_process = threading.Thread(target=data_grabbing)
learning_process = threading.Thread(target=learning)
visualization_data = threading.Thread(target=visualization)

data_process.start()
learning_process.start()
# visualization_data.start()

