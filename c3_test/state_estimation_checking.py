import lcs_class_state_dep
import lcs.optim as opt
import numpy as np
import scipy
import time
import pdb
import matplotlib.pyplot as plt

logdir = "/usr/rory-workspace/data/experiment_logs/2023/07_10_23"
log_num = "{:02}".format(0)
stateinput_file = "{}/{}/State_Input-{}.npz".format(logdir, log_num, log_num)
stateinputraw_file = "{}/{}/State_Input_Raw-{}.npz".format(logdir, log_num, log_num)

# data_stateinput: 'q': Franka joint angle; 'R_b': ball orientation quaternion; 'p_b': ball position;
# 'q_dot': Frank joint velocity; 'w_b': ball angular velocity; 'v_b': ball velocity; 'u': joint effort (torque)'
# 'timestamp_state': timestamp for the states
data_stateinput = np.load(stateinput_file)
timestamp_state = data_stateinput['timestamp_state']
p_b = data_stateinput['p_b']
v_b = data_stateinput['v_b']

data_stateinput_raw = np.load(stateinputraw_file)
timestamp_state_raw = data_stateinput_raw ['timestamp_state']
p_b_raw = data_stateinput_raw['p_b']
v_b_raw = data_stateinput_raw['v_b']

plt.plot(timestamp_state_raw,v_b_raw[2,:])
plt.plot(timestamp_state,v_b[2,:])

mdic = {"ball_position": p_b, "ball_velocity": v_b, "timestamp": timestamp_state}
scipy.io.savemat('ball_state.mat', mdic)
mdic = {"ball_position_raw": p_b_raw, "ball_velocity_raw": v_b_raw, "timestamp_raw": timestamp_state_raw}
scipy.io.savemat('ball_state_raw.mat', mdic)


plt.show()
