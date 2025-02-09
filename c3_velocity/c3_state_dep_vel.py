import lcs_class_state_dep_vel
import lcs.optim as opt
import numpy as np
import scipy
import time
import pdb
import scipy

# Modified from Wainxin's original code, modify the code to incoperate state dependent LCS matrices collected from the plant
start_time = time.time()

################################################# load data ############################################################
# x_batch: (num_data, num_state)
# u_batch: (num_data, num_input)
# res_batch: (num_data, num_state)
x_batch_list = []
u_batch_list = []
res_batch_list = []

x_model_AB_list = []
x_model_lcs_list = []
x_next_true_list = []

A_list = []
B_list = []
D_list = []
d_list = []
E_list = []
F_list = []
H_list = []
c_list = []

n_state = 19
n_control = 3
n_lam = 12
n_vel = 9

# load the data, should be improved in the future
# 100 single push data, 10 data points per push
for i in range(10):
    # The state and input data
    log_num = "{:02}".format(i)
    data_dir_xu = "/usr/rory-workspace/data/c3_learning/data_new/dt_001/State_Residual-{}.npz".format(log_num)
    data_train = np.load(data_dir_xu, allow_pickle=True)
    x_batch_raw = data_train['state_plant'][:-1,:]
    u_batch_raw = data_train['input'][:-1,:]
    res_batch_raw = data_train['residual']
    res_batch_raw = res_batch_raw[:,n_state-n_vel:n_state]
    x_batch_list.append(x_batch_raw)
    u_batch_list.append(u_batch_raw)
    res_batch_list.append(res_batch_raw)
    # also load the model predicted next state and true next state
    x_next_model_AB_raw = data_train['state_model_AB'][:-1,:]
    x_next_model_AB_raw = x_next_model_AB_raw[:,n_state-n_vel:n_state]
    x_model_AB_list.append(x_next_model_AB_raw)
    x_next_true_raw = data_train['state_plant'][1:,:]
    x_next_true_raw = x_next_true_raw[:,n_state-n_vel:n_state]
    x_next_true_list.append(x_next_true_raw)
    x_model_lcs_raw = data_train['state_model'][1:,:]
    x_model_lcs_raw = x_model_lcs_raw[:,n_state-n_vel:n_state]
    x_model_lcs_list.append(x_model_lcs_raw)

    # The system data
    data_dir_lcs = "/usr/rory-workspace/data/c3_learning/data_new/dt_001/LCS_Matrices-{}.npz".format(log_num)
    data_lcs = np.load(data_dir_lcs, allow_pickle=True)
    A = data_lcs['A_lcs'][:-1]
    A = A[:,n_state-n_vel:n_state]
    B = data_lcs['B_lcs'][:-1]
    B = B[:,n_state-n_vel:n_state]
    D = data_lcs['D_lcs'][:-1]
    D = D[:,n_state-n_vel:n_state]
    d = data_lcs['d_lcs'][:-1]
    d = d[:,n_state-n_vel:n_state]
    E = data_lcs['E_lcs'][:-1]
    F = data_lcs['F_lcs'][:-1]
    H = data_lcs['H_lcs'][:-1]
    c = data_lcs['c_lcs'][:-1]
    A_list.append(A)
    B_list.append(B)
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
x_model_lcs = np.concatenate(x_model_lcs_list,axis=0)
x_next_true = np.concatenate(x_next_true_list,axis=0)

A_batch = np.zeros((x_batch.shape[0],n_vel,n_state))
B_batch = np.zeros((x_batch.shape[0],n_vel,n_control))
D_batch = np.concatenate(D_list,axis=0)
A_batch = np.zeros((x_batch.shape[0],n_vel))
E_batch = np.concatenate(E_list,axis=0)
F_batch = np.concatenate(F_list,axis=0)
H_batch = np.concatenate(H_list,axis=0)
c_batch = np.concatenate(c_list,axis=0)

np.set_printoptions(linewidth=450)
# pdb.set_trace()
# import matplotlib.pyplot as plt
# # # briefly check the result
# # ball position and velocity
# plt.figure(figsize = (24,16))
# plt.plot(res_batch[:,2]*100, label='ee z position residual')
# plt.ylabel("position (cm)", fontsize=20)
# plt.xlabel("timestep k (every 0.1s)", fontsize=20)
# plt.legend(fontsize=20)
# plt.xticks(size = 20)
# plt.yticks(size = 20)
# #
# plt.figure(figsize = (24,16))
# plt.plot(res_batch[:,12]*100, label='ee z velocity residual')
# plt.ylabel("velocity (cm/s)", fontsize=20)
# plt.xlabel("timestep k (every 0.1s)", fontsize=20)
# plt.legend(fontsize=20)
# plt.xticks(size = 20)
# plt.yticks(size = 20)
# plt.show()
#
# plt.figure(figsize = (24,16))
# plt.plot(x_model_AB[:100,7]*100, label='x position model AB')
# plt.plot(x_model_lcs[:100,7]*100, label='x position model lcs')
# plt.plot(x_next_true[:100,7]*100, label='x position actual')
# plt.ylabel("position (cm)", fontsize=20)
# plt.xlabel("timestep k (every 0.1s)", fontsize=20)
# plt.legend(fontsize=20)
# plt.xticks(size = 20)
# plt.yticks(size = 20)
#
# plt.figure(figsize = (24,16))
# plt.plot(x_model_AB[:100,16]*100, label='x velocity model AB')
# plt.plot(x_model_lcs[:100,16]*100, label='x velocity model lcs')
# plt.plot(x_next_true[:100,16]*100, label='x velocity actual')
# plt.ylabel("velocity (cm/s)", fontsize=20)
# plt.xlabel("timestep k (every 0.1s)", fontsize=20)
# plt.legend(fontsize=20)
# plt.xticks(size = 20)
# plt.yticks(size = 20)
# plt.show()
#
# pdb.set_trace()

############################################## setting learner #########################################################
# system parameters
training_data_size = x_batch.shape[0]

# warm start options
# lcs_init = np.load("/usr/rory-workspace/data/c3_learning/data_new/dt_01/LCS_Matrices-00.npz", allow_pickle=True)
A_init = np.zeros(n_vel * n_state)
A_init_m = np.zeros((n_vel,n_state))
B_init = np.zeros(n_vel * n_control)
B_init_m = np.zeros((n_vel,n_control))
D_init = np.zeros(n_vel * n_lam)
d_init = np.zeros(n_vel)
E_init = np.zeros(n_lam * n_state)
F_init = np.zeros(n_lam * n_lam)
H_init = np.zeros(n_lam * n_control)
c_init = np.zeros(n_lam)
# c_init[2] = 0.1

# reparameterization part F divided into G and H_re
# F_init_m = lcs_init['F_lcs'][0] + 0.5 * np.eye(n_lam,n_lam) # force F to be positive definite
# F_init_m = 0.5 * np.eye(n_lam,n_lam) # force F to be positive definite
# G_init = np.linalg.cholesky((F_init_m+F_init_m.T)/2).T
# G_init_list = []
# for j in range(G_init.shape[1]):
#     G_init_list.append(G_init[0:j+1,j])
# G_init = np.concatenate(G_init_list)
# H_re_init = 0.5 * F_init_m.flatten('F')
G_init = np.zeros(int((n_lam + 1) * n_lam / 2)).flatten('F')
S_init = np.zeros((n_lam,n_lam)).flatten('F')

# There are two warm start options
# Option 1 : warm start with A,B,D,d,E,H,c and G,S, A,B,d should be near zero since our model predicts the A,B part
vn_curr_theta = np.concatenate([A_init,B_init,D_init,d_init,E_init,H_init,G_init,S_init,c_init])
# Option 2 : warm start with D,d,E,H,c and G,S only, fixe A,B,d to be zeros since we believe the normal dynamics model
# (robot properties from URDF) is nearly perfect
# vn_curr_theta = np.concatenate([D_init,E_init,H_init,G_init,S_init,c_init])

# Structure Options
# imposing the structure we want to learning, give the pre-defined matrix masks, 0 for entries want to fix, 1 for entries need to learnt

# A_mask = np.zeros((n_vel, n_state))
A_mask = np.ones((n_vel, n_state))
# A_mask[3:6,:] = np.zeros((3,n_state))

# B_mask = np.zeros((n_vel, n_control))
B_mask = np.ones((n_vel, n_control))
# B_mask[3:6,:] = np.zeros((3,n_control))

# D_mask = np.zeros((n_vel, n_lam))
D_mask = np.ones((n_vel, n_lam))
D_mask[:,0:2] = np.zeros((n_vel, 2))
# D_mask[3:6,:] = np.zeros((3,n_lam))

# d_mask = np.zeros(n_vel)
d_mask = np.ones(n_vel)

# E_mask = np.zeros((n_lam, n_state))
E_mask = np.ones((n_lam, n_state))
E_mask[0:2,:] = np.zeros((2, n_state))

# H_mask = np.zeros((n_lam, n_control))
H_mask = np.ones((n_lam, n_control))
H_mask[0:2,:] = np.zeros((2, n_control))

# c_mask = np.zeros(n_lam)
# c_mask[2] = 1
# c_mask[3] = 1
c_mask = np.ones(n_lam)
c_mask[0:2] = 0

# need to think carefully later about this part
# G_mask = np.zeros(int((n_lam + 1) * n_lam / 2))
# S_mask = np.zeros((n_lam,n_lam))
G_mask = np.ones(int((n_lam + 1) * n_lam / 2))
S_mask = np.ones((n_lam,n_lam))

# establish the violation-based learner
F_stiffness = 0.5
gamma = 1e-1
epsilon = 1e0
Q_ee = np.eye(3)
Q_br = np.eye(3)
Q_bp = np.eye(3)
Q = scipy.linalg.block_diag(Q_ee,Q_br,Q_bp)

# pdb.set_trace()
vn_learner = lcs_class_state_dep_vel.LCS_VN(n_state=n_state, n_control=n_control, n_lam=n_lam, n_vel=n_vel, F_stiffness=F_stiffness, Q=Q,
                                            A_mask=A_mask, B_mask=B_mask, D_mask=D_mask, dyn_offset_mask=d_mask, E_mask=E_mask,
                                            H_mask=H_mask, G_mask=G_mask, S_mask=S_mask, lcp_offset_mask=c_mask)
vn_learner.diff(gamma=gamma, epsilon=epsilon, w_D=1e-6, D_ref=0, w_F=0e-6, F_ref=0)
# vn_learner = lcs_class_state_dep_vel.LCS_VN(n_state=n_state, n_control=n_control, n_lam=n_lam, n_vel=n_vel, A=A_init_m, B=B_init_m,
#                                         dyn_offset=d_init, F_stiffness=F_stiffness, Q=Q, A_mask=None, B_mask=None, D_mask=D_mask,
#                                         dyn_offset_mask=None, E_mask=E_mask, H_mask=H_mask, G_mask=G_mask, S_mask=S_mask, lcp_offset_mask=c_mask)
# vn_learner.diff(gamma=gamma, epsilon=epsilon, w_D=1e-6, D_ref=0, w_F=0e-6, F_ref=0)

# establish the optimizer
vn_learning_rate = 0.8e-3
vn_optimizier = opt.Adam()
vn_optimizier.learning_rate = vn_learning_rate

# training loop
max_iter = 100
mini_batch_size = 100
# vn_curr_theta = 0.01 * np.random.randn(vn_learner.n_theta)


############################################### start learning #########################################################
total_loss_list = []
dyn_loss_list = []
lcp_loss_list = []
theta_learnt_list = []
for iter in range(max_iter):

    all_indices = np.random.permutation(training_data_size)

    for batch in range(int(np.floor(training_data_size / mini_batch_size))):
        # mini_batch_size
        # start_data_time = time.time()
        shuffle_index = all_indices[batch * mini_batch_size:(batch + 1) * mini_batch_size]
        x_mini_batch = x_batch[shuffle_index]
        u_mini_batch = u_batch[shuffle_index]
        res_mini_batch = res_batch[shuffle_index]

        # horizontally concatenate the matrices (batchsize,m,n) -> (m, batchsize*n)
        A_mini_batch = A_batch[shuffle_index]
        A_mini_batch = A_mini_batch.swapaxes(0, 1).reshape(n_vel, -1)

        B_mini_batch = B_batch[shuffle_index]
        B_mini_batch = B_mini_batch.swapaxes(0, 1).reshape(n_vel, -1)

        D_mini_batch = D_batch[shuffle_index]
        D_mini_batch = D_mini_batch.swapaxes(0, 1).reshape(n_vel, -1)

        d_mini_batch = d_batch[shuffle_index]
        d_mini_batch = d_mini_batch.T

        E_mini_batch = E_batch[shuffle_index]
        E_mini_batch = E_mini_batch.swapaxes(0, 1).reshape(n_lam, -1)

        H_mini_batch = H_batch[shuffle_index]
        H_mini_batch = H_mini_batch.swapaxes(0, 1).reshape(n_lam, -1)

        F_mini_batch = F_batch[shuffle_index]
        F_mini_batch = F_mini_batch.swapaxes(0, 1).reshape(n_lam, -1)

        c_mini_batch = c_batch[shuffle_index]
        c_mini_batch = c_mini_batch.T
        # pdb.set_trace()
        # end_data_time = time.time()
        # print("Data_time: " + str(end_data_time - start_data_time))
        # do one step for VN
        vn_mean_loss, vn_dtheta, vn_dyn_loss, vn_lcp_loss, _ = vn_learner.step(batch_x=x_mini_batch,
            batch_u=u_mini_batch, batch_x_next=res_mini_batch, current_theta=vn_curr_theta, batch_A=A_mini_batch,
            batch_B=B_mini_batch, batch_D=D_mini_batch, batch_dynamic_offset=d_mini_batch, batch_E=E_mini_batch,
            batch_H=H_mini_batch, batch_F=F_mini_batch, batch_lcp_offset=c_mini_batch)
        vn_curr_theta = vn_optimizier.step(vn_curr_theta, vn_dtheta)

    theta_learnt_list.append(vn_curr_theta)
    total_loss_list.append(vn_mean_loss)
    dyn_loss_list.append(vn_dyn_loss)
    lcp_loss_list.append(vn_lcp_loss)

    print('iter:', iter, 'vn_loss: ', vn_mean_loss, 'vn_dyn_loss: ',vn_dyn_loss, 'vn_lcp_loss',  vn_lcp_loss)


total_loss_array = np.array(total_loss_list)
best_index = np.argmin(total_loss_array)
vn_curr_theta = theta_learnt_list[best_index]

end_time = time.time()
time_spent = end_time-start_time
print(time_spent)
import matplotlib.pyplot as plt
# check the loss curve
plt.figure(figsize = (24,16))
plt.plot(total_loss_list, label='total loss')
plt.plot(dyn_loss_list, label='dynamic loss (prediction loss)')
plt.plot(lcp_loss_list, label='lcp loss (violation loss)')
plt.legend(fontsize=20)

# pdb.set_trace()
############################################### get and save results ###################################################
A_res = vn_learner.A_fn(vn_curr_theta)
B_res = vn_learner.B_fn(vn_curr_theta)
# note the notation difference between the two papers
D_res = vn_learner.D_fn(vn_curr_theta)
d_res = vn_learner.dyn_offset_fn(vn_curr_theta)
# note the notation difference between the two papers
E_res = vn_learner.E_fn(vn_curr_theta)
F_res = vn_learner.F_fn(vn_curr_theta)
H_res = vn_learner.H_fn(vn_curr_theta)
c_res = vn_learner.lcp_offset_fn(vn_curr_theta)
G_res = vn_learner.G_fn(vn_curr_theta)
S_res = vn_learner.S_fn(vn_curr_theta)


# pdb.set_trace()
# store as npy file for validation and visualization
mdic_resdyn ={"A_res":A_res,"B_res":B_res,"D_res":D_res,"d_res":d_res,"E_res":E_res,"F_res":F_res,"H_res":H_res,"c_res":
    c_res,"G_res": G_res, "S_res":S_res}
npz_file = 'data_velocity/learned_res_dyn'
np.savez(npz_file, **mdic_resdyn)


# store as seperate csv file for each matrices, integrating into c++ for sanity check
np.savetxt('data_velocity/A_res.csv', A_res, delimiter=',')
np.savetxt('data_velocity/B_res.csv', B_res, delimiter=',')
np.savetxt('data_velocity/D_res.csv', D_res, delimiter=',')
np.savetxt('data_velocity/d_res.csv', d_res, delimiter=',')
np.savetxt('data_velocity/E_res.csv', E_res, delimiter=',')
np.savetxt('data_velocity/F_res.csv', F_res, delimiter=',')
np.savetxt('data_velocity/H_res.csv', H_res, delimiter=',')
np.savetxt('data_velocity/c_res.csv', c_res, delimiter=',')
# also record the factorized part
np.savetxt('data_velocity/G_res.csv', G_res, delimiter=',')
np.savetxt('data_velocity/S_res.csv', S_res, delimiter=',')



################################# Validate the correctness of the learned dynamics #####################################
# load the result
res_dyn = np.load('data_velocity/learned_res_dyn.npz', allow_pickle=True)
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

lcs_expert = lcs_class_state_dep_vel.LCS_Gen(n_state=n_state, n_control=n_control, n_lam=n_lam, n_vel=n_vel,
                         A=A_res, B=B_res, D=D_res, dyn_offset=d_res,
                         E=E_res, H=H_res, F=F_res, lcp_offset=c_res)

A_batch = A_batch.swapaxes(0, 1).reshape(n_vel, -1)
B_batch = B_batch.swapaxes(0, 1).reshape(n_vel, -1)
D_batch = D_batch.swapaxes(0, 1).reshape(n_vel, -1)
d_batch = d_batch.T
E_batch = E_batch.swapaxes(0, 1).reshape(n_lam, -1)
H_batch = H_batch.swapaxes(0, 1).reshape(n_lam, -1)
F_batch = F_batch.swapaxes(0, 1).reshape(n_lam, -1)
c_batch = c_batch.T

res_learned, lam_learned = lcs_expert.nextState(x_batch=x_batch, u_batch=u_batch, batch_A=A_batch,
            batch_B=B_batch, batch_D=D_batch, batch_dynamic_offset=d_batch, batch_E=E_batch, batch_H=H_batch,
            batch_F=F_batch, batch_lcp_offset=c_batch)
x_next_adp = x_model_AB + res_learned
res_new = x_next_true - x_next_adp
# res_new = res_batch-res_learned
# pdb.set_trace()
import matplotlib.pyplot as plt
# briefly check the result
# ball position and velocity
plt.figure(figsize = (24,16))
plt.plot(res_batch[:,6]*100, label='x velocity residual')
plt.plot(res_learned[:,6]*100, label='x velocity residual learned')
# plt.plot(res_new[:,17]*100, label='x velocity residual after compensation')
plt.ylabel("velocity (cm/s)", fontsize=20)
plt.xlabel("timestep k (every 0.1s)", fontsize=20)
plt.legend(fontsize=20)
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.show()

plt.figure(figsize = (24,16))
plt.plot((res_batch[1:,6]-res_batch[0:-1,6])*100, label='collected')
plt.plot((res_learned[1:,6]-res_learned[0:-1,6])*100, label='learned')
plt.xlabel("timestep k (every 0.1s)", fontsize=20)
plt.legend(fontsize=20)
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.show()

plt.figure(figsize = (24,16))
plt.plot(res_batch[:,7]*100, label='y velocity residual')
plt.plot(res_learned[:,7]*100, label='y velocity residual learned')
# plt.plot(res_new[:,17]*100, label='x velocity residual after compensation')
plt.ylabel("velocity (cm/s)", fontsize=20)
plt.xlabel("timestep k (every 0.1s)", fontsize=20)
plt.legend(fontsize=20)
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.show()

plt.figure(figsize = (24,16))
plt.plot((res_batch[1:,7]-res_batch[0:-1,7])*100, label='collected')
plt.plot((res_learned[1:,7]-res_learned[0:-1,7])*100, label='learned')
plt.xlabel("timestep k (every 0.1s)", fontsize=20)
plt.legend(fontsize=20)
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.show()

plt.figure(figsize = (24,16))
plt.plot(res_batch[:,8]*100, label='z velocity residual')
plt.plot(res_learned[:,8]*100, label='z velocity residual learned')
# plt.plot(res_new[:,17]*100, label='x velocity residual after compensation')
plt.ylabel("velocity (cm/s)", fontsize=20)
plt.xlabel("timestep k (every 0.1s)", fontsize=20)
plt.legend(fontsize=20)
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.show()

plt.figure(figsize = (24,16))
plt.plot((res_batch[1:,8]-res_batch[0:-1,8])*100, label='collected')
plt.plot((res_learned[1:,8]-res_learned[0:-1,8])*100, label='learned')
plt.xlabel("timestep k (every 0.1s)", fontsize=20)
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
