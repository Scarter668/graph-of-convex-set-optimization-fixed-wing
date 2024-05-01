from pydrake.common.containers import namedview 

from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix

import numpy as np

# from pydrake.all import (
#     RotationMatrix
# )


# To test
import GeometryUtils as pyGeo 


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


def UnflattenFixedWingStatesNED(p_Dr,dp_Dr,ddp_Dr,dddp_Dr, m, g):
    
    '''
    input: x: np.array of shape (3,)
    '''
    
    p, dp, ddp, dddp = ConvertToNEDcoordinate(p_Dr,dp_Dr,ddp_Dr,dddp_Dr)
    
    
    
    
    
    
    
    s = FixedWingStatesNED(np.zeros(12))
    
    p_n = CoG(p)
    dp_n = CoG(dp)
    ddp_n = CoG(ddp)
    dddp_n = CoG(dddp)
    
    
    
    s[0:3] = p  # x, y, z
    
    s.v_a = np.sqrt( dp.T @ dp) 
    
    s.betha = 0 # assumed 0 for fixed wing
    
    s.alpha = 0 # assumed 0 for now  # np.arctan2(dx[2], dx[0])
    
    s.gamma = np.arcsin(- dp_n.z/s.v_a)
    
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
    
            
    
    return s[:]


def ExtractTransformationNED(statesNED):
    
    s = FixedWingStatesNED(statesNED)
    R_OA = RollPitchYaw(s.mu, s.gamma, s.chi).ToRotationMatrix()
    
    R_BA = RotationMatrix.MakeYRotation(-s.alpha) @ RotationMatrix.MakeZRotation(s.betha)
    
    R_OB = R_OA @ R_BA.transpose()  
    
    
    # R_OB = R_IB
    X_IB = RigidTransform(R_OB, s[0:3])
    
    return X_IB
    

def ExtractTransformation(statesNED):
    
    X_IB = ExtractTransformationNED(statesNED)
    
    
    X_DrI = RigidTransform(RotationMatrix.MakeXRotation(np.pi))
    
    X_DrB = X_DrI @ X_IB
    
    return X_DrB
