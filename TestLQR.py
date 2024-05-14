from pydrake.common.containers import namedview 

from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix

import numpy as np

from underactuated.meshcat_utils import AddMeshcatTriad

import Simulation as sim    


import GeometryUtils as geoUtils 
from pydrake.geometry import Rgba

import pickle

from pydrake.all import Simulator

    
import pydrake.symbolic as sym

from FixedWing import FullState

from pydrake.all import StartMeshcat
import FixedWing as fw


from matplotlib import pyplot as plt

from pydrake.all import PiecewisePolynomial



# from pydrake.all import (
#     RotationMatrix
# )

# Implemented in accordance with notation from https://thesis.unipd.it/retrieve/a2c7ed88-c8a9-47ad-8a4c-798b2a4c345e/Pasquali_Monika.pdf

FixedWingStatesNED = namedview(
    "FixedWingNStatesNED", ["x", "y", "z", "v_a", "betha", "alpha", "chi", "gamma", "mu", "p", "q", "r"]
)

# Center of gravity of the aircraft
CoG = namedview(
    "CoG", ["x", "y", "z"]
)


def ConvertToNEDcoordinate(p_Dr,dp_Dr,ddp_Dr,dddp_Dr):
    
    X_IDr = RigidTransform(RotationMatrix.MakeXRotation(-np.pi))
    
    p_ned = X_IDr @ p_Dr
    dp_ned = X_IDr @ dp_Dr
    ddp_ned = X_IDr @ ddp_Dr
    dddp_ned = X_IDr @ dddp_Dr
    
    return p_ned, dp_ned, ddp_ned, dddp_ned
    

ind = 0 
 
def UnflattenFixedWingStatesNED(p_Dr,dp_Dr,ddp_Dr,dddp_Dr, m, g):
    
    '''
    input: x: np.array of shape (3,)
    '''
    global ind
    p, dp, ddp, dddp = ConvertToNEDcoordinate(p_Dr,dp_Dr,ddp_Dr,dddp_Dr)
    
    
    
    # geoUtils.visualize_point(meshcat, p, label=f"test/p_{ind}", radius=0.01, color=Rgba(.06, 0, 0, 1)) 
    ind +=1
    
    s = FixedWingStatesNED(np.zeros(12))
    
    p_n = CoG(p)
    dp_n = CoG(dp)
    ddp_n = CoG(ddp)
    dddp_n = CoG(dddp)
    
    
    
    s[0:3] = p  # x, y, z
    
    s.v_a = np.sqrt( dp.T @ dp) 
    
    s.betha = 0  #np.arctan2(-dp_n.y, dp_n.x) #np.pi/6 # assumed 0 for fixed wing
    
    s.alpha = 0 # np.arctan2(dp_n.z , dp_n.x) # assumed 0 for now  # np.arctan2(dx[2], dx[0])
    
    s.gamma = np.arcsin(- dp_n.z/s.v_a)
    
    # Works better with atan2(ddp_n.y, ddp_n.x) hmm
    s.chi = np.arctan2(dp_n.y, dp_n.x)
    
    
    
    dx_ddx = dp_n.x * ddp_n.x
    dy_ddy = dp_n.y * ddp_n.y
    dz_ddz = dp_n.z * ddp_n.z
    sum_dt_p = dx_ddx + dy_ddy + dz_ddz
    
    
    dv_a = sum_dt_p / s.v_a # PAPER says s.v_a**2, but I think it should be s.v_a
    
    
    d_dx_ddx = ddp_n.x**2 + dp_n.x*dddp_n.x
    d_dy_ddy = ddp_n.y**2 + dp_n.y*dddp_n.y
    d_dz_ddz = ddp_n.z**2 + dp_n.z*dddp_n.z
    
    sum_ddt_p = d_dx_ddx + d_dy_ddy + d_dz_ddz
    
    ddv_a = (sum_ddt_p*s.v_a - sum_dt_p * dv_a ) / s.v_a**2  # PAPER says something else, but I think it might be a propagation error from dv_a
    
    sqrt_exp = np.sqrt(1 - (dp_n.z/s.v_a)**2) 
    dgamma = - ddp_n.z/(s.v_a * sqrt_exp) + dp_n.z*dv_a / (s.v_a**2 * sqrt_exp) # SAME AS PAPER
    
    f = dp_n.x * ddp_n.y - dp_n.y * ddp_n.x
    g = dp_n.x**2 + dp_n.y**2
    dchi = f / g # SAME AS PAPER
    
    df = dddp_n.y*dp_n.x - dddp_n.x*dp_n.y 
    dg = 2*dp_n.x*ddp_n.x + 2*dp_n.y*ddp_n.y
    ddchi = (df*g - f*dg ) / g**2 # SAME AS PAPER
    
    
    t = s.v_a * m * np.cos(s.gamma) * dchi
    b = s.v_a * m * dgamma + np.cos(s.gamma)*g*m
    s.mu = np.arctan2(t, b) # FROM PAPER
    
    # print(f"z_dot: {dp_n.z}")
    # print(f"s.v_a: {s.v_a}, sqrt_exp: {sqrt_exp}, s.b : {b}")
    
            
    
    return s[:]



def ExtractTransformationNED(statesNED):
    
    s = FixedWingStatesNED(statesNED)
    
    # TODO : iplement rotation by hand
    
    # print out the euler angles 
    # print(f"chi: {s.chi}, gamma: {s.gamma}, mu: {s.mu}")
    
    R_OA = RollPitchYaw(s.mu, s.gamma, s.chi).ToRotationMatrix()
    
    R_BA = RotationMatrix.MakeYRotation(-s.alpha) @ RotationMatrix.MakeZRotation(s.betha)
    
    R_OB = R_OA @ R_BA.transpose()  
    
    # print(f"s.chi: {s.chi}, s.gamma: {s.gamma}, s.mu: {s.mu}")
    print("R_OB: ", R_OB)
    
    # R_OB_temp = R_OB @ RotationMatrix.MakeXRotation(np.pi/2)
    # R_OB = R_IB
    X_IB = RigidTransform(R_OB, s[0:3])


    
    return X_IB 
    

def ExtractTransformation(statesNED):
    
    X_IB = ExtractTransformationNED(statesNED)
    
    
    X_DrI = RigidTransform(RotationMatrix.MakeXRotation(np.pi))
    
    X_DrB = X_DrI @ X_IB
    
    return X_DrB



t_arr = np.array([sym.Variable("t")])

def f_straigth_x(t_arr):
    m = sym if t_arr.dtype == object else np
    
    return np.array(
        [ t_arr[0],
         0,
         1
        ]
    )

def f_straight_xy(t_arr):
    m = sym if t_arr.dtype == object else np
    return np.array(
        [ t_arr[0],
          -t_arr[0],
          1
        ]
    )

def f_straight_xz(t_arr):
    m = sym if t_arr.dtype == object else np
    return np.array(
        [ t_arr[0],
          0,
          t_arr[0]*.1
        ]
    )

def f_sine_xz(t_arr):
    m = sym if t_arr.dtype == object else np
    return np.array(
        [ t_arr[0],
          0,
          1.2 + m.sin(t_arr[0])
        ]
    )
    
def f_sine_xy(t_arr):
    m = sym if t_arr.dtype == object else np
    return np.array(
        [ t_arr[0],
          m.sin(t_arr[0])*0.5,
          1,
        ]
    )
    
def f_sine_xyz(t_arr):
    m = sym if t_arr.dtype == object else np
    return np.array(
        [ t_arr[0],
          t_arr[0],
          1.2 + m.sin(t_arr[0])
        ]
    )


def f_circle(t_arr):
    m = sym if t_arr.dtype == object else np
    return np.array(
        [ m.cos(t_arr[0]),
          m.sin(t_arr[0]),
          1
        ]
    )

def f_infinity(t_arr):
    m = sym if t_arr.dtype == object else np
    return 2*np.array(
        [ m.cos(t_arr[0]) / (1 + m.sin(t)**2),
         m.cos(t_arr[0]) * m.sin(t_arr[0]) / (1 + m.sin(t_arr[0])**2),
         1 
        ]
    )


i = 0

f = None
if i==0:
    f = f_straigth_x
elif i==1:
    f = f_straight_xy
elif i==2:
    f = f_straight_xz
elif i==3:
    f = f_sine_xy
elif i==4:
    f = f_sine_xz
elif i==5:
    f = f_sine_xyz
elif i==6:
    f = f_circle
elif i==7:
    f = f_infinity



f_t = f(t_arr)
print("f_t: ", f_t)

df_t = sym.Jacobian(f_t, t_arr).ravel() 
print("df_t: ", df_t)

ddf_t = sym.Jacobian(df_t,t_arr).ravel()
print("ddf_t: ", ddf_t)

dddf_t = sym.Jacobian(ddf_t,t_arr).ravel()
print("dddf_t: ", dddf_t)


def get_pathAndDerivativs(t):
    env = {t_arr[0] : t}
    p = sym.Evaluate(f_t, env).ravel()
    dp = sym.Evaluate(df_t, env).ravel()
    ddp = sym.Evaluate(ddf_t, env).ravel()
    dddp = sym.Evaluate(dddf_t, env).ravel()
    
    
    return p, dp, ddp, dddp



m = 1.56
g = 9.81

# meshcat.Delete("traj_source/")
# meshcat.Delete("test/")


start_time = 0
end_time = 20

num_timesteps = 1000
num_dofs = 3
p_numeric = np.empty((num_timesteps, num_dofs))
dp_numeric = np.empty((num_timesteps, num_dofs))
ddp_numeric = np.empty((num_timesteps, num_dofs))
dddp_numeric = np.empty((num_timesteps, num_dofs))


sample_times_s = np.linspace(
    start_time, end_time, num=num_timesteps, endpoint=True
)
for i, t in enumerate(sample_times_s):
    ( p_numeric[i],
        dp_numeric[i], 
        ddp_numeric[i], 
        dddp_numeric[i] ) = get_pathAndDerivativs(t)


x_states = []
u_states = []

for p, dp, ddp, dddp, t in zip(
    p_numeric,
    dp_numeric,
    ddp_numeric,
    dddp_numeric,
    sample_times_s
):

    fullState = FullState(np.zeros(16))
    stateNED = UnflattenFixedWingStatesNED(p, dp, ddp, dddp, m, g)
    
    
    
    fullState[:12] = stateNED
    
    fullState.delta_le = -.1#np.sin(t)
    fullState.delta_re = -.1#np.sin(t)
    fullState.delta_rm = .5#30*np.pi*t
    fullState.delta_lm = .5#30*np.pi*t
    
    
    x_states.append(fullState[:])
    u_states.append([-.1,-0.1, .5, .5])
    
    
    



start_time = 0
end_time = 20 

# traj = 1


# Have to define trajectory for x and u



Q = np.diag([10, 10, 10, 1, 1, 1, 1])
Q = np.diag([10, 10, 10, 10, 10, 10, 10, 10, 10,
             1, 1, 1, 1, 1, 1, 1])
R = np.diag([0.1, 0.1, 0.1, 0.1])
Qf = np.diag(
    [
        (1 / 0.05) ** 2,
        (1 / 0.05) ** 2,
        (1 / 0.05) ** 2,
        (1 / 0.05) ** 2,
        (1 / 0.05) ** 2,
        (1 / 0.05) ** 2,
        (1 / 0.05) ** 2,
        (1 / 0.05) ** 2,
        (1 / 0.05) ** 2,
        (1 / 3.0) ** 2,
        (1 / 3.0) ** 2,
        (1 / 3.0) ** 2,
        (1 / 3.0) ** 2,
        (1 / 3.0) ** 2,
        (1 / 3.0) ** 2,
        (1 / 3.0) ** 2,
    ]
)
       
x_traj = PiecewisePolynomial.CubicShapePreserving(sample_times_s, np.array(x_states).T, zero_end_point_derivatives=False)
u_traj = PiecewisePolynomial.CubicShapePreserving(sample_times_s, np.array(u_states).T, zero_end_point_derivatives=False)


# meshcat = StartMeshcat()

# x = np.zeros(16)
# fw = fw.draw_glider(x, meshcat)


simEnv = sim.SimulationEnvironment()

simEnv.connect_meshcat()
simEnv.add_fixed_wing()


# input = np.array([-0.5 ,-0.5, .5, .5])
# simEnv.add_constant_inputSource(input)
simEnv.add_controller(x_traj, u_traj, Q, R, Qf)



simEnv.build_model()
simEnv.save_and_display_diagram(save=False)


meshcat = simEnv.meshcat
diagram = simEnv.diagram

plane_plant = simEnv.fixed_plane
meshcat_visualizer = simEnv.visualizer