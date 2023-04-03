import lcs_class
import lcs.optim as opt
import numpy as np
import scipy
import time
import pdb

# Modified from Wainxin's original code, a rough test for learning LCS for C3
st = time.time()

# load data
# x_batch: (num_data, num_state)
# u_batch: (num_data, num_input)
# res_batch: (num_data, num_state)
x_batch_list = []
u_batch_list = []
res_batch_list = []

x_model_AB_list = []
x_next_true_list = []

# load the data, should be improved in the futrue
# 20 single push data, 99 data points per push
for i in range(20):
    log_num = "{:02}".format(i)
    data_dir = "data_raw/State_Residual-{}.npz".format(log_num)
    data_train = np.load(data_dir, allow_pickle=True)
    # Lcs_plant = np.load('data_raw/LCS_Matrices-01.npz', allow_pickle=True)
    x_batch_raw = data_train['state_plant'][:-1,:]
    u_batch_raw = data_train['input'][:-1,:]
    res_batch_raw = data_train['residual']
    x_batch_list.append(x_batch_raw)
    u_batch_list.append(u_batch_raw)
    res_batch_list.append(res_batch_raw)
    # also load the model predicted next state and true next state
    x_next_model_AB_raw = data_train['state_model_AB'][:-1,:]
    x_model_AB_list.append(x_next_model_AB_raw)
    x_next_true_raw = data_train['state_plant'][1:,:]
    x_next_true_list.append(x_next_true_raw)

x_batch = np.concatenate(x_batch_list,axis=0)
u_batch = np.concatenate(u_batch_list,axis=0)
res_batch = np.concatenate(res_batch_list,axis=0)

x_model_AB = np.concatenate(x_model_AB_list,axis=0)
x_next_true = np.concatenate(x_next_true_list,axis=0)

# import matplotlib.pyplot as plt
# # briefly check the result
# # ball position and velocity
# plt.figure(figsize = (24,16))
# plt.plot(x_batch[:,16]*100, label='x velocity actual')
# plt.ylabel("velocity (cm/s)", fontsize=20)
# plt.xlabel("timestep k (every 0.01s)", fontsize=20)
# plt.legend(fontsize=20)
# plt.xticks(size = 20)
# plt.yticks(size = 20)
# plt.show()


# system parameters
n_state = 19
n_control = 3
n_lam = 12
training_data_size = x_batch.shape[0]


# warm start options
lcs_init = np.load("data_raw/LCS_Matrices-00.npz", allow_pickle=True)
A_init = lcs_init['A_lcs'][0].flatten('F')
B_init = lcs_init['B_lcs'][0].flatten('F')
# note the notation difference between the two papers D->C
D_init = lcs_init['D_lcs'][0].flatten('F')
d_init = lcs_init['d_lcs'][0].flatten('F')
# note the notation difference between the two papers E->D; H->E
E_init = lcs_init['E_lcs'][0].flatten('F')
F_init = lcs_init['F_lcs'][0].flatten('F')
H_init = lcs_init['H_lcs'][0].flatten('F')
c_init = lcs_init['c_lcs'][0].flatten('F')

# reparameterization part F divided into G and H_re
F_init_m = lcs_init['F_lcs'][0] + 0.5 * np.eye(n_lam,n_lam) # force F to be positive definite
G_init = np.linalg.cholesky((F_init_m+F_init_m.T)/2).T
G_init_list = []
for j in range(G_init.shape[1]):
    G_init_list.append(G_init[0:j+1,j])
G_init = np.concatenate(G_init_list)
H_re_init = 0.5 * F_init_m.flatten('F')

# pdb.set_trace()

# warm start with A,B,D,d,E,H,c and G,H_re
vn_curr_theta = np.concatenate([A_init,B_init,D_init,d_init,E_init,H_init,G_init,H_re_init,c_init])

# establish the VN learner (violation-based method)
F_stiffness = 1
gamma = 1e-1
epsilon = 1e0
vn_learner = lcs_class.LCS_VN(n_state=n_state, n_control=n_control, n_lam=n_lam, F_stiffness=F_stiffness)
vn_learner.diff(gamma=gamma, epsilon=epsilon, w_C=1e-6, C_ref=0, w_F=0e-6, F_ref=0)

# establish the optimizer
vn_learning_rate = 0.1e-3
vn_optimizier = opt.Adam()
vn_optimizier.learning_rate = vn_learning_rate

# training loop
max_iter = 60
mini_batch_size = 50
# vn_curr_theta = 0.01 * np.random.randn(vn_learner.n_theta)


# for iter in range(max_iter):
#
#     all_indices = np.random.permutation(training_data_size)
#
#     for batch in range(int(np.floor(training_data_size / mini_batch_size))):
#         # mini_batch_size
#         shuffle_index = all_indices[batch * mini_batch_size:(batch + 1) * mini_batch_size]
#         x_mini_batch = x_batch[shuffle_index]
#         u_mini_batch = u_batch[shuffle_index]
#         res_mini_batch = res_batch[shuffle_index]
#
#         # do one step for VN
#         vn_mean_loss, vn_dtheta, vn_dyn_loss, vn_lcp_loss, _ = vn_learner.step(batch_x=x_mini_batch,
#                                                                                batch_u=u_mini_batch,
#                                                                                batch_x_next=res_mini_batch,
#                                                                                current_theta=vn_curr_theta)
#         vn_curr_theta = vn_optimizier.step(vn_curr_theta, vn_dtheta)
#
#     print('iter:', iter, 'vn_loss: ', vn_mean_loss)
#
# A_res = vn_learner.A_fn(vn_curr_theta)
# B_res = vn_learner.B_fn(vn_curr_theta)
# # note the notation difference between the two papers
# D_res = vn_learner.C_fn(vn_curr_theta)
# d_res = vn_learner.dyn_offset_fn(vn_curr_theta)
# # note the notation difference between the two papers
# E_res = vn_learner.D_fn(vn_curr_theta)
# F_res = vn_learner.F_fn(vn_curr_theta)
# H_res = vn_learner.E_fn(vn_curr_theta)
# c_res = vn_learner.lcp_offset_fn(vn_curr_theta)
#
#
# mdic_resdyn ={"A_res":A_res,"B_res":B_res,"D_res":D_res,"d_res":d_res,"E_res":E_res,"F_res":F_res,"H_res":H_res,"c_res":c_res}
# npz_file = 'learned_res_dyn'
# np.savez(npz_file, **mdic_resdyn)



res_dyn = np.load('learned_res_dyn.npz', allow_pickle=True)
A_res = res_dyn['A_res']
B_res = res_dyn['B_res']
# note the notation difference between the two papers
D_res = res_dyn['D_res']
d_res = res_dyn['d_res']
# note the notation difference between the two papers
E_res = res_dyn['E_res']
F_res = res_dyn['F_res']
H_res = res_dyn['H_res']
c_res = res_dyn['c_res']


# --------------------------- Validate the correctness of the learned dynamics ---------------------------------- #
lcs_expert = lcs_class.LCS_Gen(n_state=n_state, n_control=n_control, n_lam=n_lam,
                         A=A_res, B=B_res, C=D_res, dyn_offset=d_res,
                         D=E_res, E=H_res, F=F_res, lcp_offset=c_res)
res_learned, lam_learned = lcs_expert.nextState(x_batch=x_batch, u_batch=u_batch)
x_next_adp = x_model_AB + res_learned
res_new = x_next_true - x_next_adp
# res_new = res_batch-res_learned

import matplotlib.pyplot as plt
# briefly check the result
# ball position and velocity
plt.figure(figsize = (24,16))
plt.plot(res_batch[:,10]*100, label='x velocity residual')
plt.plot(res_learned[:,10]*100, label='x velocity residual learned')
# plt.plot(res_new[:,17]*100, label='x velocity residual after compensation')
plt.ylabel("velocity (cm/s)", fontsize=20)
plt.xlabel("timestep k (every 0.01s)", fontsize=20)
plt.legend(fontsize=20)
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.show()

plt.figure(figsize = (24,16))
plt.plot(res_batch[:,7]*100, label='x position residual')
plt.plot(res_learned[:,7]*100, label='x position residual learned')
# plt.plot(res_new[:,17]*100, label='x velocity residual after compensation')
plt.ylabel("position (cm)", fontsize=20)
plt.xlabel("timestep k (every 0.01s)", fontsize=20)
plt.legend(fontsize=20)
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.show()

# plt.figure(figsize = (24,16))
# plt.plot(lam_learned[:,1], label='contact force')
# plt.ylabel("force (N)", fontsize=20)
# plt.xlabel("timestep k (every 0.01s)", fontsize=20)
# plt.legend(fontsize=20)
# plt.xticks(size = 20)
# plt.yticks(size = 20)
# plt.show()
