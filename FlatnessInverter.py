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
    "FixedWingNStatesNED", ["x", "y", "z", "v_a", "beta", "alpha", "chi", "gamma", "mu", "p", "q", "r"]
)

# Center of gravity of the aircraft
CoG = namedview(
    "CoG", ["x", "y", "z"]
)


def ConvertToNEDcoordinate(p_Dr,dp_Dr,ddp_Dr,dddp_Dr, ddddp_Dr):
    
    X_IDr = RigidTransform(RotationMatrix.MakeXRotation(-np.pi))
    
    p_ned = X_IDr @ p_Dr
    dp_ned = X_IDr @ dp_Dr
    ddp_ned = X_IDr @ ddp_Dr
    dddp_ned = X_IDr @ dddp_Dr
    ddddp_ned = X_IDr @ ddddp_Dr
    
    return p_ned, dp_ned, ddp_ned, dddp_ned, ddddp_ned


def UnflattenFixedWingStatesNED(p_Dr,dp_Dr,ddp_Dr,dddp_Dr, ddddp_Dr, m, g):
    
    '''
    input: x: np.array of shape (3,)
    '''
    
    p, dp, ddp, dddp, ddddp = ConvertToNEDcoordinate(p_Dr,dp_Dr,ddp_Dr,dddp_Dr, ddddp_Dr)
    
    
    
    
    
    
    
    s = FixedWingStatesNED(np.zeros(12))
    
    p_n = CoG(p)
    dp_n = CoG(dp)
    ddp_n = CoG(ddp)
    dddp_n = CoG(dddp)
    ddddp_n = CoG(ddddp)
    
    
    
    s[0:3] = p  # x, y, z
    
    s.v_a = np.sqrt( dp.T @ dp) 
    
    s.beta = 0 #np.arctan2(dp_n.z, dp_n.x) # assumed 0 for fixed wing
    
    s.alpha = 0 #np.arctan2(dp_n.y, s.v_a) # assumed 0 for now  # np.arctan2(dx[2], dx[0])
    
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
    
    
    
    ## Calculate ddgamma
    
    # Calculation of u and its derivatives
    u = -dp_n.z / s.v_a
    u_prime = -ddp_n.z / s.v_a + dp_n.z * dv_a / s.v_a**2
    u_double_prime = (-dddp_n.z * s.v_a - ddp_n.z * dv_a) / s.v_a**2 - \
                    (ddp_n.z * s.v_a - dp_n.z * dv_a) * dv_a / s.v_a**3 + \
                    2 * dp_n.z * dv_a**2 / s.v_a**4

    # Calculate the denominator sqrt(1 - u^2) and its higher power for stability in the formula
    sqrt_1_minus_u_squared = np.sqrt(1 - u**2)
    cubed_sqrt_1_minus_u_squared = (1 - u**2)**(1.5)

    # Calculating dgamma and ddgamma
    dgamma = -ddp_n.z / (s.v_a * sqrt_1_minus_u_squared) + dp_n.z * dv_a / (s.v_a**2 * sqrt_1_minus_u_squared)

    ddgamma = u_double_prime / sqrt_1_minus_u_squared + u * u_prime**2 / cubed_sqrt_1_minus_u_squared

    
    # sqrt_exp = np.sqrt(1 - (dp_n.z/s.v_a)**2) 
    # dgamma = - ddp_n.z/(s.v_a * sqrt_exp) + dp_n.z*dv_a / (s.v_a**2 * sqrt_exp) # SAME AS PAPER
    
    
    
    
    f = dp_n.x * ddp_n.y - dp_n.y * ddp_n.x
    g = dp_n.x**2 + dp_n.y**2
    dchi = f / g # SAME AS PAPER
    
    df = dddp_n.y*dp_n.x - dddp_n.x*dp_n.y 
    dg = 2*dp_n.x*ddp_n.x + 2*dp_n.y*ddp_n.y
    ddchi = (df*g - f*dg ) / g**2 # SAME AS PAPER
    
    dddf = dp_n.x * ddddp_n.y - dp_n.y * ddddp_n.x + ddp_n.x * dddp_n.y - ddp_n.y * dddp_n.x
    dddg = 2 * (dp_n.x * dddp_n.x + ddp_n.x * ddp_n.x + dp_n.y * dddp_n.y + ddp_n.y * ddp_n.y)

    # Compute dddchi
    dddchi = ((dddf * g - df * dddg) / g**2) - 2 * ((df * g - f * dg) / g**2) * ((dddg * g - dg * dg) / g**3)


    t = s.v_a * m * np.cos(s.gamma) * dchi
    b = s.v_a * m * dgamma + np.cos(s.gamma)*g*m
    s.mu = np.arctan2(t, b) # FROM PAPER
    

    # First derivatives of t and b
    dt = m * (dv_a * np.cos(s.gamma) * dchi + s.v_a * (-np.sin(s.gamma) * dgamma) * dchi + s.v_a * np.cos(s.gamma) * ddchi)
    db = m * (dv_a * dgamma + s.v_a * ddgamma - np.sin(s.gamma) * dgamma * g)

    # dmu calculation
    dmu = (b * dt - t * db) / (b**2 + t**2)

    
    ddt = m * (ddv_a * np.cos(s.gamma) * dchi + 2 * dv_a * (-np.sin(s.gamma) * dgamma) * dchi +
           dv_a * np.cos(s.gamma) * ddchi + s.v_a * (-np.cos(s.gamma) * dgamma**2 - np.sin(s.gamma) * ddgamma) * dchi +
           2 * s.v_a * (-np.sin(s.gamma) * dgamma) * ddchi + s.v_a * np.cos(s.gamma) * dddchi)
    ddb = m * (ddv_a * dgamma + 2 * dv_a * ddgamma - np.cos(s.gamma) * dgamma**2 * g - np.sin(s.gamma) * ddgamma * g)

    # Update ddmu based on the quotient rule applied to the dmu formula
    ddmu = ((b * ddt + db * dt) * (b**2 + t**2) - (b * dt - t * db) * 2 * (b * db + t * dt)) / (b**2 + t**2)**2

    
    
    Z_a = - (s.v_a * m * dgamma + np.cos(s.gamma) * g * m) / (np.cos(s.mu))
    
    
    dZ_a = - (m* dgamma * dv_a + m * ddgamma * s.v_a - g*m* np.sin(s.gamma)*dgamma) / (np.cos(s.mu)) - np.sin(s.mu)*(m* s.v_a * dgamma + g * m * np.cos(s.gamma)) * dmu / np.cos(s.mu)**2
    
    X_a = m * s.v_a  + np.sin(s.gamma) * g
    
    dalpha = 0 # assumed 0 angular attack acceleration
    
    s.q = dalpha - np.cos(s.gamma)*np.cos(s.mu)*g / s.v_a - Z_a / (- s.v_a *m) #  PAPER
    
    s.p = 1 /( 1 + np.tan(s.alpha)**2) * ( dmu/np.cos(s.alpha)  
                                          - np.tan(s.alpha)*(np.cos(s.gamma)*np.sin(s.mu)*g / (np.cos(s.alpha)* s.v_a) ) 
                                          + Z_a*np.sin(s.mu)*np.sin(s.gamma)/(m*np.cos(s.gamma)*s.v_a*np.cos(s.alpha))    ) # PAPER
    
    s.r = (s.v_a * np.sin(s.alpha)*s.p + np.cos(s.gamma)*np.sin(s.mu)*g) / (s.v_a * np.cos(s.alpha)) # PAPER
    
    dp_ = (1 / (s.v_a**2 * m * np.cos(s.gamma)**2)) * (
        -s.v_a * np.cos(s.gamma) * (g * np.cos(s.gamma)**2 * np.sin(s.mu) * np.cos(s.alpha) * m +
                            np.sin(s.alpha) * m * s.v_a * dmu * np.cos(s.gamma) +
                            np.sin(s.gamma) * np.sin(s.mu) * np.sin(s.alpha) * Z_a) * dalpha 
        - np.cos(s.gamma) * np.cos(s.mu) * s.v_a * (g * np.cos(s.gamma)**2 * np.sin(s.alpha) * m -
                                        np.sin(s.gamma) * np.cos(s.alpha) * Z_a) * dmu 
        + np.cos(s.gamma) * np.sin(s.mu) * (g * np.cos(s.gamma)**2 * np.sin(s.alpha) * m -
                                    np.sin(s.gamma) * np.cos(s.alpha) * Z_a) * dv_a 
        + (np.sin(s.mu) * (g * np.cos(s.gamma)**2 * np.sin(s.gamma) * np.sin(s.alpha) * m +
                    np.cos(s.alpha) * Z_a) * dgamma +  np.cos(s.gamma) * np.cos(s.alpha) *
                    (np.cos(s.gamma) * m * ddmu * s.v_a + np.sin(s.mu) * np.sin(s.gamma) * dZ_a)) * s.v_a)
    
    
    ddalpha = 0 # assumed 0 angular attack acceleration
    dq = ( (np.cos(s.gamma)*np.cos(s.mu)* g*m + Z_a)* dv_a 
          + s.v_a*(g*np.sin(s.gamma)*np.cos(s.mu)*dgamma*m + g*np.sin(s.mu)*m*np.cos(s.gamma)*dmu + m*s.v_a*ddalpha - dZ_a )
        ) / (m * s.v_a**2)
    
    
    # Calculate each term
    cos_alpha = np.cos(s.alpha)
    cos_gamma = np.cos(s.gamma)
    sin_alpha = np.sin(s.alpha)
    sin_gamma = np.sin(s.gamma)
    sin_mu = np.sin(s.mu)
    term1 = s.v_a * (g * sin_mu * sin_alpha * cos_gamma + s.v_a * s.p) * dalpha
    term2 = ((s.v_a * sin_alpha * s.p + cos_gamma * sin_mu * g) * dv_a) 
    term3 = s.v_a * (dgamma * sin_gamma * sin_mu * g - cos_gamma * dmu * sin_mu * g - s.v_a * sin_alpha * dp_)


    # Full expression for dr
    dr = (term1 / (cos_alpha**2 * s.v_a**2)) - cos_alpha * (term2  + term3) 
    
    
    sdot = FixedWingStatesNED(np.zeros(12))
    
    sdot[0:3] = dp
    sdot.v_a = dv_a
    sdot.beta = 0
    sdot.alpha = 0
    sdot.gamma = dgamma
    sdot.chi = dchi
    sdot.mu = dmu
    sdot.p = dp_
    sdot.q = dq
    sdot.r = dr
    
       
    
    
    

    
    return s[:]


def ExtractTransformationNED(statesNED):
    
    R_OB = Extract_R_OB(statesNED)  
    
    if type(R_OB) == np.ndarray:
        X_IB = RigidTransform(RotationMatrix(R_OB), statesNED[0:3])
    
    else:
        X_IB = RigidTransform(R_OB, statesNED[0:3])
    
    return X_IB
    

def ExtractTransformation(statesNED):
    
    X_IB = ExtractTransformationNED(statesNED)
    
    
    X_DrI = RigidTransform(RotationMatrix.MakeXRotation(np.pi))
    
    X_DrB = X_DrI @ X_IB
    
    return X_DrB


def Extract_R_BA(statesNED):
    
    s = FixedWingStatesNED(statesNED)
    
    if type(s.alpha) != float:
        R_BA = Ry(-s.alpha) @ Rz(s.beta)
        # print("used non drake rotation")
    else:
        R_BA = RotationMatrix.MakeYRotation(-s.alpha) @ RotationMatrix.MakeZRotation(s.beta)
    
    return R_BA

def Extract_R_OA(statesNed):
        
    s = FixedWingStatesNED(statesNed)
    
    if type(s.mu) != float:
        R_OA = Rz(s.chi) @ Ry(s.gamma) @ Rx(s.mu)
        # print("used non drake rotation")
    
    else:
        R_OA = RollPitchYaw(s.mu, s.gamma, s.chi).ToRotationMatrix()
    
    return R_OA

def Extract_R_OB(statesNED):
    
    R_OA = Extract_R_OA(statesNED)
    
    R_BA = Extract_R_BA(statesNED)
    
    R_OB = R_OA @ R_BA.transpose()
    
    return R_OB



def Rz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

def Ry(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])
def Rx(theta):
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])
