import lcs_class_state_dep
import lcs.optim as opt
import numpy as np
import scipy
import time
import pdb

# Modified from Wainxin's original code, modify the code to incoperate state dependent LCS matrices collected from the plant
st = time.time()

################################################# load data ############################################################
# x_batch: (num_data, num_state)
# u_batch: (num_data, num_input)
# res_batch: (num_data, num_state)
x_batch_list = []
u_batch_list = []
res_batch_list = []

x_model_AB_list = []
x_next_true_list = []

D_list = []
d_list = []
E_list = []
F_list = []
H_list = []
c_list = []

# load the data, should be improved in the futrue
# 20 single push data, 99 data points per push
for i in range(20):
    # The state and input data
    log_num = "{:02}".format(i)
    data_dir_xu = "data_raw/State_Residual-{}.npz".format(log_num)
    data_train = np.load(data_dir_xu, allow_pickle=True)
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

    # The system data
    data_dir_lcs = "data_raw/LCS_Matrices-{}.npz".format(log_num)
    data_lcs = np.load(data_dir_lcs, allow_pickle=True)
    D = data_lcs['D_lcs'][:-1]
    d = data_lcs['d_lcs'][:-1]
    E = data_lcs['E_lcs'][:-1]
    F = data_lcs['F_lcs'][:-1]
    H = data_lcs['H_lcs'][:-1]
    c = data_lcs['c_lcs'][:-1]
    D_list.append(D)
    d_list.append(d)
    E_list.append(E)
    H_list.append(H)
    F_list.append(F)
    c_list.append(c)

# concatenate as numpy array
x_batch = np.concatenate(x_batch_list,axis=0)
u_batch = np.concatenate(u_batch_list,axis=0)
res_batch = np.concatenate(res_batch_list,axis=0)

x_model_AB = np.concatenate(x_model_AB_list,axis=0)
x_next_true = np.concatenate(x_next_true_list,axis=0)

A_batch = np.zeros((1980,19,19))
B_batch = np.zeros((1980,19,3))
D_batch = np.concatenate(D_list,axis=0)
d_batch = np.concatenate(d_list,axis=0)
E_batch = np.concatenate(E_list,axis=0)
F_batch = np.concatenate(F_list,axis=0)
H_batch = np.concatenate(H_list,axis=0)
c_batch = np.concatenate(c_list,axis=0)


# pdb.set_trace()

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

############################################## setting learner #########################################################
# system parameters
n_state = 19
n_control = 3
n_lam = 12
training_data_size = x_batch.shape[0]

# warm start options
lcs_init = np.load("data_raw/LCS_Matrices-00.npz", allow_pickle=True)
A_init = np.zeros(n_state * n_state)
A_init_m = np.zeros((n_state,n_state))
B_init = np.zeros(n_state * n_control)
B_init_m = np.zeros((n_state,n_control))
# note the notation difference between the two papers D->C
D_init = np.zeros(n_state * n_lam)
d_init = np.zeros(n_state)
# note the notation difference between the two papers E->D; H->E
E_init = np.zeros(n_lam * n_state)
F_init = np.zeros(n_lam * n_lam)
H_init = np.zeros(n_lam * n_control)
c_init = np.zeros(n_lam)

# lcs_init = np.load("data_raw/LCS_Matrices-00.npz", allow_pickle=True)
# A_init = np.zeros(n_state * n_state)
# A_init_m = np.zeros((n_state,n_state))
# B_init = np.zeros(n_state * n_control)
# B_init_m = np.zeros((n_state,n_control))
# # note the notation difference between the two papers D->C
# D_init = lcs_init['D_lcs'][0].flatten('F')
# d_init = np.zeros(n_state)
# # note the notation difference between the two papers E->D; H->E
# E_init = lcs_init['E_lcs'][0].flatten('F')
# F_init = lcs_init['F_lcs'][0].flatten('F')
# H_init = lcs_init['H_lcs'][0].flatten('F')
# c_init = lcs_init['c_lcs'][0].flatten('F')

# reparameterization part F divided into G and H_re
# F_init_m = lcs_init['F_lcs'][0] + 0.5 * np.eye(n_lam,n_lam) # force F to be positive definite
F_init_m = 0.5 * np.eye(n_lam,n_lam) # force F to be positive definite
G_init = np.linalg.cholesky((F_init_m+F_init_m.T)/2).T
G_init_list = []
for j in range(G_init.shape[1]):
    G_init_list.append(G_init[0:j+1,j])
G_init = np.concatenate(G_init_list)
H_re_init = 0.5 * F_init_m.flatten('F')

# There are two warm start options
# Option 1 : warm start with A,B,D,d,E,H,c and G,H_re, A,B should be near zero since our model predicts the A,B part
# vn_curr_theta = np.concatenate([A_init,B_init,D_init,d_init,E_init,H_init,G_init,H_re_init,c_init])

# Option 2 : warm start with D,d,E,H,c and G,H_re only, fixe A,B to be zeros since we believe the normal dynamics model
# (robot properties from URDF) is nearly perfect
vn_curr_theta = np.concatenate([D_init,E_init,H_init,G_init,H_re_init,c_init])

# establish the VN learner (violation-based method)
F_stiffness = 0.5
gamma = 1e-1
epsilon = 5e-2
# vn_learner = lcs_class_state_dep.LCS_VN(n_state=n_state, n_control=n_control, n_lam=n_lam, F_stiffness=F_stiffness)
vn_learner = lcs_class_state_dep.LCS_VN(n_state=n_state, n_control=n_control, n_lam=n_lam, A=A_init_m, B=B_init_m,
                                        dyn_offset=d_init, F_stiffness=F_stiffness)
vn_learner.diff(gamma=gamma, epsilon=epsilon, w_C=1e-6, C_ref=0, w_F=0e-6, F_ref=0)

# establish the optimizer
vn_learning_rate = 0.1e-3
vn_optimizier = opt.Adam()
vn_optimizier.learning_rate = vn_learning_rate

# training loop
max_iter = 50
mini_batch_size = 50
# vn_curr_theta = 0.01 * np.random.randn(vn_learner.n_theta)

############################################### start learning #########################################################
for iter in range(max_iter):

    all_indices = np.random.permutation(training_data_size)

    for batch in range(int(np.floor(training_data_size / mini_batch_size))):
        # mini_batch_size
        shuffle_index = all_indices[batch * mini_batch_size:(batch + 1) * mini_batch_size]
        x_mini_batch = x_batch[shuffle_index]
        u_mini_batch = u_batch[shuffle_index]
        res_mini_batch = res_batch[shuffle_index]

        # do one step for VN, note the difference of notation in two papers
        # note the notation difference between the two papers (alp)->(wanxin), D->C; E->D; H->E;
        vn_mean_loss, vn_dtheta, vn_dyn_loss, vn_lcp_loss, _ = vn_learner.step(batch_x=x_mini_batch,
            batch_u=u_mini_batch, batch_x_next=res_mini_batch, current_theta=vn_curr_theta, batch_A=A_batch,
            batch_B=B_batch, batch_C=D_batch, batch_dynamic_offset=d_batch, batch_D=E_batch, batch_E=H_batch,
            batch_F=F_batch, batch_lcp_offset=c_batch)
        vn_curr_theta = vn_optimizier.step(vn_curr_theta, vn_dtheta)

    print('iter:', iter, 'vn_loss: ', vn_mean_loss)


############################################### get and save results ###################################################
A_res = vn_learner.A_fn(vn_curr_theta)
B_res = vn_learner.B_fn(vn_curr_theta)
# note the notation difference between the two papers
D_res = vn_learner.C_fn(vn_curr_theta)
d_res = vn_learner.dyn_offset_fn(vn_curr_theta)
# note the notation difference between the two papers
E_res = vn_learner.D_fn(vn_curr_theta)
F_res = vn_learner.F_fn(vn_curr_theta)
H_res = vn_learner.E_fn(vn_curr_theta)
c_res = vn_learner.lcp_offset_fn(vn_curr_theta)


# store as npy file for validation and visualization
mdic_resdyn ={"A_res":A_res,"B_res":B_res,"D_res":D_res,"d_res":d_res,"E_res":E_res,"F_res":F_res,"H_res":H_res,"c_res":c_res}
npz_file = 'data_state_dep/learned_res_dyn'
np.savez(npz_file, **mdic_resdyn)


# store as seperate csv file for each matrices, integrating into c++ for sanity check
np.savetxt('data_state_dep/A_res.csv', A_res, delimiter=',')
np.savetxt('data_state_dep/B_res.csv', B_res, delimiter=',')
np.savetxt('data_state_dep/D_res.csv', D_res, delimiter=',')
np.savetxt('data_state_dep/d_res.csv', d_res, delimiter=',')
np.savetxt('data_state_dep/E_res.csv', E_res, delimiter=',')
np.savetxt('data_state_dep/F_res.csv', F_res, delimiter=',')
np.savetxt('data_state_dep/H_res.csv', H_res, delimiter=',')
np.savetxt('data_state_dep/c_res.csv', c_res, delimiter=',')


################################# Validate the correctness of the learned dynamics #####################################
# load the result
res_dyn = np.load('data_state_dep/learned_res_dyn.npz', allow_pickle=True)
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

lcs_expert = lcs_class_state_dep.LCS_Gen(n_state=n_state, n_control=n_control, n_lam=n_lam,
                         A=A_res, B=B_res, C=D_res, dyn_offset=d_res,
                         D=E_res, E=H_res, F=F_res, lcp_offset=c_res)
res_learned, lam_learned = lcs_expert.nextState(x_batch=x_batch, u_batch=u_batch,batch_A=A_batch,
            batch_B=B_batch, batch_C=D_batch, batch_dynamic_offset=d_batch, batch_D=E_batch, batch_E=H_batch,
            batch_F=F_batch, batch_lcp_offset=c_batch)
x_next_adp = x_model_AB + res_learned
res_new = x_next_true - x_next_adp
# res_new = res_batch-res_learned
# pdb.set_trace()
import matplotlib.pyplot as plt
# briefly check the result
# ball position and velocity
plt.figure(figsize = (24,16))
plt.plot(res_batch[:,17]*100, label='x velocity residual')
plt.plot(res_learned[:,17]*100, label='x velocity residual learned')
# plt.plot(res_new[:,17]*100, label='x velocity residual after compensation')
plt.ylabel("velocity (cm/s)", fontsize=20)
plt.xlabel("timestep k (every 0.01s)", fontsize=20)
plt.legend(fontsize=20)
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.show()

plt.figure(figsize = (24,16))
plt.plot((res_batch[1:,17]-res_batch[0:-1,17])*100, label='collected')
plt.plot((res_learned[1:,17]-res_learned[0:-1,17])*100, label='learned')
plt.xlabel("timestep k (every 0.01s)", fontsize=20)
plt.legend(fontsize=20)
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.show()

# pdb.set_trace()

# plt.figure(figsize = (24,16))
# plt.plot(res_batch[:,7]*100, label='x position residual')
# plt.plot(res_learned[:,7]*100, label='x position residual learned')
# # plt.plot(res_new[:,17]*100, label='x velocity residual after compensation')
# plt.ylabel("position (cm)", fontsize=20)
# plt.xlabel("timestep k (every 0.01s)", fontsize=20)
# plt.legend(fontsize=20)
# plt.xticks(size = 20)
# plt.yticks(size = 20)
# plt.show()

# plt.figure(figsize = (24,16))
# plt.plot(lam_learned[:,3], label='contact force')
# plt.ylabel("force (N)", fontsize=20)
# plt.xlabel("timestep k (every 0.01s)", fontsize=20)
# plt.legend(fontsize=20)
# plt.xticks(size = 20)
# plt.yticks(size = 20)
# plt.show()
