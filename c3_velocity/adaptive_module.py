import sys

# hack the sys path to let it be able to be called from procman
# print the sys path when using pycharm and record the sys path as below
# print (sys.path)
sys.path = ['/usr/rory-workspace/Learning-LCS/c3_velocity', '/usr/rory-workspace/Learning-LCS', '/usr/lib/python38.zip', '/usr/lib/python3.8', '/usr/lib/python3.8/lib-dynload', '/usr/rory-workspace/Learning-LCS/venv/lib/python3.8/site-packages', '/usr/rory-workspace/Learning-LCS']

import lcs_class_state_dep_vel
import lcs.optim as opt
import numpy as np
import scipy
import time
import pdb
import lcm
from lcm_types.lcm_adaptive import lcmt_lcs, lcmt_learning_data, lcmt_visual
import threading
import matplotlib.pyplot as plt

############################################## data setting #########################################################
num_state = 19
num_velocity = 9
num_control = 3
num_lambda = 8
utime = 0

# initialization of the residual lcs, all to be zeros and can be first published
A_res = np.zeros((num_velocity, num_state))
B_res = np.zeros((num_velocity, num_control))
D_res = np.zeros((num_velocity, num_lambda))
d_res = np.zeros(num_velocity)

E_res = np.zeros((num_lambda, num_state))
F_res = np.zeros((num_lambda, num_lambda))
H_res = np.zeros((num_lambda, num_control))
c_res = np.zeros(num_lambda)
# c_res = 0.04 * np.ones(num_lambda)

# initialization of the checking and visualization data
c_grad = np.zeros(num_lambda)
d_grad = np.zeros(num_velocity)

dyn_error_check = np.zeros(num_velocity)
lcp_error_check = np.zeros(num_lambda)
lambda_check = np.zeros(num_lambda)
res_check= np.zeros(num_velocity)
lambda_n_check = np.zeros(2)
Dlambda_check = np.zeros(num_velocity)

total_loss_check = np.zeros(1)
dyn_loss_check = np.zeros(1)
lcp_loss_check = np.zeros(1)
period_loss_check = np.zeros(1)
period_dyn_loss_check = np.zeros(1)
period_lcp_loss_check = np.zeros(1)

############################################## LCM and data buffer setting #############################################
# define a data buffer class that keeps taking in the data and is growing
# currently through keep appending the list, might run out of the cache
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
        # initialization of the settling time
        self.settling_time = 0

    def learning_data_receiver(self, channel, data):
        # decode message
        msg = lcmt_learning_data.decode(data)

        # if not reaching settling time (have nor started), then don't grab data
        safety_dt = 0.05
        if msg.utime * 1e-6 < msg.settling_time + safety_dt:
            return
        else:
            # grab data and store in the predefined list
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

# create and initialize the data buffer for further use
lcm_data = LCM_data_buffer()
# subscribe the channel that contains the learning data set
subscription_learning_dataset = lcm_data.lc.subscribe("LEARNING_DATASET", lcm_data.learning_data_receiver)
# prepare to lcm publish object, one is to publish the learned lcs
# the other is used to publish the data that needs to visualize for sanity check
lc_publish = lcm.LCM()
lc_visual_publish = lcm.LCM()

############################################## Learner setting #########################################################
cnt = 0
period_cnt = 0

# warm start options
A_init = np.zeros(num_velocity * num_state)
B_init = np.zeros(num_velocity * num_control)
D_init = np.zeros(num_velocity * num_lambda)
d_init = np.zeros(num_velocity)

E_init = np.zeros(num_lambda * num_state)
F_init = np.zeros(num_lambda * num_lambda)
H_init = np.zeros(num_lambda * num_control)
c_init = np.zeros(num_lambda)

# reparameterization part F divided into G (upper triangular) and S (skew symmetric)
G_init = np.zeros(int((num_lambda + 1) * num_lambda / 2))
S_init = np.zeros(num_lambda * num_lambda)
vn_curr_theta = np.concatenate([A_init,B_init,D_init,d_init,E_init,H_init,G_init,S_init,c_init])

# mask option, enforcing the learning structure, the masked entry (masked with 0) would remain the initial value.

A_mask = np.zeros((num_velocity, num_state))
B_mask = np.zeros((num_velocity, num_control))
D_mask = np.zeros((num_velocity, num_lambda))
d_mask = np.zeros(num_velocity)
# d_mask = np.ones(num_velocity)

E_mask = np.zeros((num_lambda, num_state))
H_mask = np.zeros((num_lambda, num_control))
c_mask = np.ones(num_lambda)

# need to carefully think about this one
G_mask = np.zeros(int((num_lambda + 1) * num_lambda / 2))
S_mask = np.zeros((num_lambda,num_lambda))

# establish the violation-based learner, the main parameters are shown below
# F_Stiffness enforce the F matrix to be PD (orginally can only guarantee PSD), need to tune
F_stiffness = 1e-1
# gamma should set to be gamma < F_Stiffness
gamma = 1e-2
# epsilon represent the weight put on LCP violation part (weight is 1/epsilon)
epsilon = 5e-4


# also assign weight for each residual, now we only learn from ball velocity residuals
Q_ee = 0 * np.eye(3)
# Q_ee = 1e-5 * np.diag(np.array([1, 1, 1]))
Q_br = 0 * np.eye(3)
# Q_br = np.diag(np.array([1, 1, 1]))
Q_bp = 1e5 * np.diag(np.array([1, 1, 1]))
Q = scipy.linalg.block_diag(Q_ee, Q_br, Q_bp)

# initialize the learning, warning would rise if dimension does not match
vn_learner = lcs_class_state_dep_vel.LCS_VN(n_state=num_state, n_control=num_control, n_lam=num_lambda, n_vel=num_velocity,
                                            F_stiffness=F_stiffness, Q=Q, A_mask=A_mask, B_mask=B_mask, D_mask=D_mask,
                                            dyn_offset_mask=d_mask, E_mask=E_mask, H_mask=H_mask, G_mask=G_mask, S_mask=S_mask,
                                            lcp_offset_mask=c_mask)
vn_learner.diff(gamma=gamma, epsilon=epsilon, w_D=0e-6, D_ref=0, w_F=0e-6, F_ref=0)

# establish the optimizer, currently choose the Adam gradient descent method
vn_learning_rate = 1e-2
vn_optimizier = opt.Adam()
vn_optimizier.learning_rate = vn_learning_rate

# currently, collect data at a fixed frequency, so the timestamp alignment can be done through index search
data_dt = 0.005
reference_model_dt = 0.1
index_span = int(reference_model_dt / data_dt)

# training loop settings
# mini_batch is done by batch computing in casadi, the batch corresponds to the time period of data_dt * mini_batch_size
mini_batch_size = 10
# max_iter is the time to do gradient steps for this batch data
max_iter = 1

# period setting
period_time = 1.5 # in the future, try directly read from yaml file (need to be careful about path)
period_threshold = period_time / (data_dt * mini_batch_size)

# gradient buffer setting, the gradient update would use the average of the gradient buffer to update to ensure that the
# contact prediction information is included
# time span of the gradient buffer, should be determined by analyzing the gradient distribution
# buffer_time = 1
# # buffer size should be determined by buffer_time, max_iter, mini_batch_size, data_dt
# buffer_size = buffer_time * max_iter / (mini_batch_size * data_dt)
# grad_buffer = np.zeros((vn_learner.n_theta, int(buffer_size)))

# storing the loss in the list for offline analysis
total_loss_list = []
dyn_loss_list = []
lcp_loss_list = []
theta_learnt_list = []
np.set_printoptions(linewidth=450)

# # for visualization in the meeting
# c_grad_list = []
# d_grad_list = []
# dyn_error_check_list = []
# lcp_error_check_list = []
# lambda_check_list = []
# lambda_n_check_list= []



########################################## multiple thread #############################################################
# use multiple threads parallel to keep grabbing data while not hindering the learning
# the above defined variables should be declared as global in the following function, check this first when debugging!
def data_grabbing():
    # data grabbing function
    global lcm_data
    while True:
        lcm_data.lc.handle()

def learning():
    global cnt, period_cnt, period_threshold
    global num_state, num_velocity, num_control, num_lambda, utime
    global A_res, B_res, D_res, d_res, E_res, F_res, H_res, c_res
    global max_iter, mini_batch_size, Q
    global vn_learner, vn_optimizier, vn_curr_theta
    global lcm_data, lc_publish, lc_visual_publish
    global total_loss_list, dyn_loss_list, lcp_loss_list, theta_learnt_list
    global grad_buffer, buffer_size
    global c_grad, d_grad, dyn_error_check, lcp_error_check, lambda_check, res_check, lambda_n_check, Dlambda_check
    global total_loss_check, dyn_loss_check, lcp_loss_check, period_loss_check, period_dyn_loss_check, period_lcp_loss_check

    while True:
        if len(lcm_data.state_list) < (cnt+1)*mini_batch_size + index_span:
            # when data enough, start learning, or just send current result
            # if learning not start, send zero matrices and regard the time as 0
            pass
        else:
            start_time = time.time()
            # get batch data (horizon data_dt * mini_batch_size) and data in the future (reference_model_dt)
            u_list = []
            x_pred_list = []
            x_batch = np.array(lcm_data.state_list[cnt*mini_batch_size: (cnt+1)*mini_batch_size])

            # 2023.8.2 use the avergae of the input over the time horizon as the prediction
            for i in range(x_batch.shape[0]):
                u = np.array(lcm_data.input_list[cnt*mini_batch_size + i : cnt*mini_batch_size + index_span + i])
                u_i = np.mean(u, axis=0)
                A_i = lcm_data.A_list[cnt * mini_batch_size + i]
                B_i = lcm_data.B_list[cnt * mini_batch_size + i]
                d_i = lcm_data.d_list[cnt * mini_batch_size + i]

                x_pred_i = A_i @ x_batch[i,:] + B_i @ u_i + d_i
                u_list.append(u_i)
                x_pred_list.append(x_pred_i)

            # u_batch = np.array(lcm_data.input_list[cnt*mini_batch_size: (cnt+1)*mini_batch_size])   # original one
            u_batch = np.array(u_list)
            # x_pred_batch = np.array(lcm_data.state_pred_list[cnt*mini_batch_size: (cnt+1)*mini_batch_size])  # original one
            x_pred_batch = np.array(x_pred_list)

            # Dlambda_check = np.mean(x_pred_batch, axis=0)

            x_next_batch = np.array(lcm_data.state_list[cnt*mini_batch_size + index_span: (cnt+1)*mini_batch_size + index_span])
            res_batch = x_next_batch[:,num_state-num_velocity:] - x_pred_batch
            res_check = np.mean(res_batch, axis=0)

            A_batch = np.zeros((num_velocity, mini_batch_size * num_state))
            B_batch = np.zeros((num_velocity, mini_batch_size * num_control))
            D_batch = np.array(lcm_data.D_list[cnt*mini_batch_size: (cnt+1)*mini_batch_size]).swapaxes(0, 1).reshape(num_velocity, -1)
            d_batch = np.zeros((num_velocity, mini_batch_size))


            E_batch = np.array(lcm_data.E_list[cnt*mini_batch_size: (cnt+1)*mini_batch_size]).swapaxes(0, 1).reshape(num_lambda, -1)
            F_batch = np.array(lcm_data.F_list[cnt*mini_batch_size: (cnt+1)*mini_batch_size]).swapaxes(0, 1).reshape(num_lambda, -1)
            H_batch = np.array(lcm_data.H_list[cnt*mini_batch_size: (cnt+1)*mini_batch_size]).swapaxes(0, 1).reshape(num_lambda, -1)
            c_batch = np.array(lcm_data.c_list[cnt*mini_batch_size: (cnt+1)*mini_batch_size]).T
            # print(np.array(
            #     lcm_data.timestamp_list[cnt * mini_batch_size + index_span: (cnt + 1) * mini_batch_size + index_span]) \
            #               -np.array(lcm_data.timestamp_list[cnt*mini_batch_size: (cnt+1)*mini_batch_size]))

            end_data_time = time.time()

            for iter in range(max_iter):
                vn_mean_loss, vn_dtheta, vn_dyn_loss, vn_lcp_loss, lam_batch, dyn_error_batch, lcp_error_batch, Dlambda_mean = \
                    vn_learner.step(batch_x=x_batch, batch_u=u_batch, batch_x_next=res_batch, current_theta=vn_curr_theta,
                                    batch_A=A_batch, batch_B=B_batch, batch_D=D_batch, batch_dynamic_offset=d_batch,
                                    batch_E=E_batch, batch_H=H_batch, batch_F=F_batch, batch_lcp_offset=c_batch)

                vn_curr_theta = vn_optimizier.step(vn_curr_theta, vn_dtheta) # comment this if do not want learning

                # data for sanity checking
                c_grad = vn_learner.lcp_offset_fn(vn_dtheta).full()
                d_grad = vn_learner.dyn_offset_fn(vn_dtheta).full()
                dyn_error_check = np.mean(dyn_error_batch, axis=0)
                lcp_error_check = np.mean(lcp_error_batch, axis=0)
                lambda_check = np.mean(lam_batch, axis=0)
                lambda_n_check[0] = np.sum(lambda_check[0:4])
                lambda_n_check[1] = np.sum(lambda_check[4:])
                Dlambda_check = Dlambda_mean
                # Dlambda_check[0:8] = Dlambda_mean
                # Dlambda_check[8] = 0


            # store loss
            # theta_learnt_list.append(vn_curr_theta)
            total_loss_list.append(vn_mean_loss.copy())
            dyn_loss_list.append(vn_dyn_loss.copy())
            lcp_loss_list.append(vn_lcp_loss.copy())
            total_loss_check = vn_mean_loss
            dyn_loss_check = vn_dyn_loss
            lcp_loss_check = vn_lcp_loss

            # print('iter:', iter, 'vn_loss: ', vn_mean_loss, 'vn_dyn_loss: ', vn_dyn_loss, 'vn_lcp_loss', vn_lcp_loss)

            # convert the learnt parameters into matrices, comment this if do not want learning
            Start_LCSdata_time = time.time()
            A_res = vn_learner.A_fn(vn_curr_theta).full()
            B_res = vn_learner.B_fn(vn_curr_theta).full()
            D_res = vn_learner.D_fn(vn_curr_theta).full()
            d_res = vn_learner.dyn_offset_fn(vn_curr_theta).full()
            E_res = vn_learner.E_fn(vn_curr_theta).full()
            F_res = vn_learner.F_fn(vn_curr_theta).full()
            H_res = vn_learner.H_fn(vn_curr_theta).full()
            c_res = vn_learner.lcp_offset_fn(vn_curr_theta).full()

            # assign the time for this calculation to be the time of the initial point of the batch data
            utime = int(lcm_data.timestamp_list[cnt*mini_batch_size] * 1e6)

            # counter add on to move to next data batch
            cnt = cnt + 1

            period_cnt = period_cnt + 1
            if period_cnt == period_threshold:
                period_cnt = 0
                period_loss_check = sum(total_loss_list) / len(total_loss_list)
                total_loss_list = []
                period_dyn_loss_check = sum(dyn_loss_list) / len(dyn_loss_list)
                dyn_loss_list = []
                period_lcp_loss_check = sum(lcp_loss_list) / len(lcp_loss_list)
                lcp_loss_list = []

            # keep publishing the data
            msg = lcmt_lcs()
            msg_visual = lcmt_visual()

            # for learned matrices publishing
            msg.num_state = num_state
            msg.num_velocity = num_velocity
            msg.num_control = num_control
            msg.num_lambda = num_lambda
            msg.utime = utime

            msg.A = A_res
            msg.B = B_res
            msg.D = D_res
            msg.d = d_res

            msg.E = E_res
            msg.F = F_res
            msg.H = H_res
            msg.c = c_res

            # for visualization and sanity check
            msg_visual.num_velocity = num_velocity
            msg_visual.num_lambda = num_lambda
            msg_visual.utime = utime

            msg_visual.d_grad = d_grad
            msg_visual.c_grad = c_grad

            msg_visual.dyn_error_check = dyn_error_check
            msg_visual.lcp_error_check = lcp_error_check
            msg_visual.lambda_check = lambda_check
            msg_visual.res_check = res_check
            msg_visual.lambda_n = lambda_n_check
            msg_visual.Dlambda_check = Dlambda_check

            msg_visual.total_loss = total_loss_check
            msg_visual.dyn_loss = dyn_loss_check
            msg_visual.lcp_loss = lcp_loss_check
            msg_visual.period_loss = period_loss_check
            msg_visual.period_dyn_loss = period_dyn_loss_check
            msg_visual.period_lcp_loss = period_lcp_loss_check

            # publish the message
            lc_publish.publish("RESIDUAL_LCS", msg.encode())
            lc_visual_publish.publish("DATA_CHECKING", msg_visual.encode())

            # end_time = time.time()
            # print(end_time - start_time)


        # check the time for each loop
        # print(cnt)
        # print("Learning Time：", end_time - start_time)
        # print("Data Time：", end_data_time - start_time + end_time - Start_LCSdata_time)
        # print(cnt*mini_batch_size)
        # print((cnt+1)*mini_batch_size)

def visualization():
    global cnt, buffer_size
    global mini_batch_size
    global lcm_data
    global total_loss_list, dyn_loss_list, lcp_loss_list, theta_learnt_list


# send the initial message
msg = lcmt_lcs()
msg_visual = lcmt_visual()

# for learned matrices publishing
msg.num_state = num_state
msg.num_velocity = num_velocity
msg.num_control = num_control
msg.num_lambda = num_lambda
msg.utime = utime

msg.A = A_res
msg.B = B_res
msg.D = D_res
msg.d = d_res
msg.E = E_res
msg.F = F_res
msg.H = H_res
msg.c = c_res

# for visualization and sanity check
msg_visual.num_velocity = num_velocity
msg_visual.num_lambda = num_lambda
msg_visual.utime = utime

msg_visual.d_grad = d_grad
msg_visual.c_grad = c_grad

msg_visual.dyn_error_check = dyn_error_check
msg_visual.lcp_error_check = lcp_error_check
msg_visual.lambda_check = lambda_check
msg_visual.res_check = res_check
msg_visual.lambda_n = lambda_n_check
msg_visual.Dlambda_check = Dlambda_check

msg_visual.total_loss = total_loss_check
msg_visual.dyn_loss = dyn_loss_check
msg_visual.lcp_loss = lcp_loss_check
msg_visual.period_loss = period_loss_check
msg_visual.period_dyn_loss = period_dyn_loss_check
msg_visual.period_lcp_loss = period_lcp_loss_check

# publish the message
lc_publish.publish("RESIDUAL_LCS", msg.encode())
lc_visual_publish.publish("DATA_CHECKING", msg_visual.encode())



# main part, excute multiple threads to run the learning and data grabbing (somtimes + plotting) at the same time
data_process = threading.Thread(target=data_grabbing)
learning_process = threading.Thread(target=learning)
# visualization_data = threading.Thread(target=visualization)

data_process.start()
learning_process.start()
# visualization_data.start()


