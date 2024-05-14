import numpy as np

from pydrake.common.containers import namedview
from pydrake.common.value import AbstractValue

from pydrake.systems.scalar_conversion import TemplateSystem
from pydrake.systems.framework import DiagramBuilder, LeafSystem

from pydrake.solvers import Solve

from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser


from pydrake.planning import DirectCollocation

from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix

from pydrake.geometry import MeshcatVisualizer, SceneGraph, Cylinder, FramePoseVector

from pydrake.trajectories import PiecewisePolynomial

from underactuated.scenarios import AddShape
from underactuated import ConfigureParser

from pydrake.all import (
    LeafSystem_,
    
)

import os


from manipulation.scenarios import AddRgbdSensors

import pydrake.math as pyMath


import FlatnessInverter as fi


from scipy.special import huber

NUM_STATES = 12
NUM_INPUTS = 4



FullState = namedview(
    "FullState",
    [
        "x", "y", "z", "v_a", "beta", "alpha", "chi", "gamma", "mu", "p", "q", "r","delta_le", "delta_re", "delta_lm", "delta_rm"
    ]
)


# Note: In order to use the Python system with drake's autodiff features, we
# have to add a little "TemplateSystem" boilerplate (for now).  For details,
# see https://drake.mit.edu/pydrake/pydrake.systems.scalar_conversion.html

@TemplateSystem.define("FixedWingPlant_")
def FixedWingPlant_(T):
    class Impl(LeafSystem_[T]):
        def _construct(self, converter=None):
            LeafSystem_[T].__init__(self, converter)
            # one inputs (elevator_velocity)
            self.DeclareVectorInputPort("inputs", NUM_INPUTS)
            self.DeclareContinuousState(NUM_STATES+NUM_INPUTS)
            # seven outputs (full state)
            self.DeclareVectorOutputPort("full_state", NUM_STATES+NUM_INPUTS, self.CopyStateOut)
            self.DeclareVectorOutputPort("spatial_force", 6, self.OutputForces)

            # Constants and parameters
            self.m = 1.56  # (kg) full weight of the delta-wing
            # self.m = 5.5
            self.g = 9.80665 # (m/sˆ2) gravitational acceleration
            self.Ixx = 0.1147 # (kg*mˆ2) (Zagi flying wing)
            self.Iyy = 0.0576 # (kg*mˆ2) (Zagi flying wing)
            self.Izz = 0.1712 # (kg*mˆ2) (Zagi flying wing)
            self.Ixz = 0.0015 # (kg*mˆ2) (Zagi flying wing)
            
            
            self.rho = 1.225 # (kg/mˆ3) (density of the air)
            self.mu = 1.81e-5 # (kg/m/s) (dynamic viscosity of the air at 15 )
            self.nu = self.mu / self.rho # (mˆ2/s) (kinematic viscosity of the air at 15 )
            
            self.S = 0.2589 # (mˆ2) surface of the delta-wing
            self.c = 0.3302 # (m) mean chord
            self.b = 1.4224 # (m) tip to tip length of the wings
            self.l_motor = self.b / 4 # distance from motor to Oxb axis
            self.S_prop = 0.0314 # (mˆ2)
            self.k_motor = 40
            self.k_T_p = 1e-6 # maximum 1 Nm torque
            self.k_omega = 1e3 # maximum 2000 rad/s
            
            

    
        def _construct_copy(self, other, converter=None):
            Impl._construct(self, converter=converter)
            


        def ComputeForces(self, context):
            fullstate = FullState(
                np.array(context.get_mutable_continuous_state_vector().CopyToVector(), copy=True)
            )
            
            u = self.EvalVectorInput(context, 0).CopyToVector()
            s = fullstate[:NUM_STATES]
            
            F, M = self.forces_moments(s[:], u)
            
            return F, M

        def DoCalcTimeDerivatives(self, context, derivatives):
            fullstate = FullState(
                np.array(context.get_mutable_continuous_state_vector().CopyToVector(),copy=True)
            )
            
            fullstate.z = pyMath.min(fullstate.z, -.1)
            
            # print(f"1 -fullstate.z: {fullstate.z}")
            u = self.EvalVectorInput(context, 0).CopyToVector()
            s = fi.FixedWingStatesNED(np.array(fullstate[:NUM_STATES], copy=True))
          
            # print(f"2 -fullstate.z: {fullstate.z}")
            s.z = pyMath.min(s.z, -.5)
            
            
            (delta_le, delta_re, w_lm, w_rm) = u
            
            # delta_le = pyMath.max(pyMath.min(delta_le, -1.1), 1.1)
            # delta_re = pyMath.max(pyMath.min(delta_re, -1.1), 1.1)
            
            u = np.array([delta_le, delta_re, w_lm, w_rm])
            
            
            # print(f"deltas: {delta_le} {delta_re} {w_lm} {w_rm}")
    
            
            # print(f"3 - fullstate.z: {fullstate.z}")
            M, F = self.forces_moments(s[:], u)
            X_a = F[0]*1
            Y_a = F[1]*1
            Z_a = F[2]*1
            L_b = M[0]*1
            M_b = M[1]*1
            N_b = M[2]*1
            
            max_force = 2000
            max_torque = 1000
            X_a = pyMath.max(pyMath.min(X_a, max_force), -max_force)
            Y_a = pyMath.max(pyMath.min(Y_a, max_force), -max_force)
            Z_a = pyMath.max(pyMath.min(Z_a, max_force), -max_force)
            L_b = pyMath.max(pyMath.min(L_b, max_torque), -max_torque)
            M_b = pyMath.max(pyMath.min(M_b, max_torque), -max_torque)
            N_b = pyMath.max(pyMath.min(N_b, max_torque), -max_torque)
            
            # print(f"X_a: {X_a} \nY_a: {Y_a} \nZ_a: {Z_a} \nL_b: {L_b} \nM_b: {M_b} \nN_b: {N_b}")
            
            fullstate_dot = FullState(np.array(fullstate[:], copy=True) )
            # print(f"4 -fullstate.z: {fullstate.z}")
            
            fullstate_dot.x = pyMath.cos(s.chi) * pyMath.cos(s.gamma) * s.v_a
            # print(f"5 -fullstate.z: {fullstate.z}")
            
            fullstate_dot.y = pyMath.sin(s.chi) * pyMath.cos(s.gamma) * s.v_a
            # print(f"6 -fullstate.z: {fullstate.z}")
            
            fullstate_dot.z = -pyMath.sin(s.gamma) * s.v_a
            # print(f"7 -fullstate.z: {fullstate.z}")
            
            fullstate_dot.v_a = X_a/self.m - pyMath.sin(s.gamma) * self.g
            
            # print(f"final -fullstate.z: {fullstate.z}")
            
            fullstate_dot.beta =  pyMath.sin(s.alpha)*s.p - pyMath.cos(s.alpha)*s.r + (pyMath.cos(s.gamma)*pyMath.sin(s.mu)*self.m*self.g + Y_a )/(self.m*s.v_a)
            fullstate_dot.alpha = s.q - (pyMath.cos(s.alpha)*s.p + pyMath.sin(s.alpha)*s.r)*pyMath.tan(s.beta) +  (pyMath.cos(s.gamma)*pyMath.cos(s.mu)*self.g)/(pyMath.cos(s.beta)*s.v_a) + Z_a/(pyMath.cos(s.beta)*s.v_a*self.m)
            
            fullstate_dot.chi = (- Z_a * pyMath.sin(s.mu) + Y_a * pyMath.cos(s.mu)) / (s.v_a * self.m * (1-0*pyMath.cos(s.mu)))
            fullstate_dot.gamma = (-pyMath.cos(s.gamma) * self.g * self.m - Y_a * pyMath.sin(s.mu) - Z_a * pyMath.cos(s.mu)) / (s.v_a * self.m)
            
            mu_dot_num1 = -pyMath.cos(s.mu) * pyMath.cos(s.gamma) * pyMath.sin(s.beta) * self.g
            mu_dot_num2 = s.p * pyMath.cos(s.alpha) + s.r * pyMath.sin(s.alpha)
            mu_dot_num3 = Z_a * pyMath.sin(s.beta)
            mu_dot_num4 =pyMath.sin(s.gamma) * (Y_a * pyMath.cos(s.mu) - Z_a * pyMath.sin(s.mu))

            # The corresponding denominators
            denom1 = s.v_a * pyMath.cos(s.beta)
            denom2 = pyMath.cos(s.beta)
            denom3 = s.v_a *self.m * pyMath.cos(s.beta)
            denom4 = s.v_a * self.m * pyMath.cos(s.gamma)
            
            fullstate_dot.mu = (mu_dot_num1 / denom1) + (mu_dot_num2 / denom2) - (mu_dot_num3 / denom3) + (mu_dot_num4 / denom4)       
            
            fullstate_dot.p = (
                (self.Ixz * (self.Ixx - self.Iyy + self.Izz) * s.p -
                (self.Ixz**2 - self.Izz *(self.Iyy - self.Izz)) * s.r) *s.q +
                self.Ixz * N_b + self.Izz * L_b) / (self.Ixx * self.Izz - self.Ixz**2)
            
            fullstate_dot.q = ( -self.Ixz * s.p**2 - s.r * (self.Ixx - self.Izz) * s.p +
                                self.Ixz * s.r**2 + M_b) / self.Iyy
            
            fullstate_dot.r = ( ((self.Ixz**2 + self.Ixx * (self.Ixx - self.Iyy)) * s.p 
            -self.Ixz * (self.Ixx - self.Iyy + self.Izz) * s.r ) * s.q + 
            self.Ixx * N_b + self.Ixz * L_b) / (self.Ixx * self.Izz - self.Ixz**2)
            # lowpassfilter with time constant 0.01
            lmbda = 10
            fullstate_dot.delta_le = lmbda*(-fullstate.delta_le) + delta_le
            fullstate_dot.delta_re = lmbda*(-fullstate.delta_re)+ delta_re
            fullstate_dot.delta_lm = w_lm
            fullstate_dot.delta_rm = w_rm
        
            
            
            # derivatives.get_mutable_vector().SetFromVector(fullstate_dot[:])
            
            # print("derivatives updated")
            # print("fullstate: ", fullstate[:])
            
            # fullstate.z = pyMath.min(fullstate.z, -.5)
            # print full statete in a nice format
            # print(f"\nx: {fullstate.x} \ty: {fullstate.y} \tz: {fullstate.z} \tv_a: {fullstate.v_a} \nbeta: {fullstate.beta} \t\talpha: {fullstate.alpha} \nchi: {fullstate.chi} \t\tgamma: {fullstate.gamma} \t\tmu: {fullstate.mu} \np: {fullstate.p} \t\tq: {fullstate.q} \t\tr: {fullstate.r} \ndelta_le: {fullstate.delta_le} \t\tdelta_re: {fullstate.delta_re} \t\tdelta_lm: {fullstate.delta_lm} \t\tdelta_rm: {fullstate.delta_rm}")
                    
        def forces_moments(self, x, u):
            # Unpack control inputs and state variables
            delta_el, delta_er, w_l, w_r = u
            state = fi.FixedWingStatesNED(np.array(x, copy=True))
            
            # delta_el = pyMath.max(pyMath.min(delta_el, -1.1), 1.1)
            # delta_er = pyMath.max(pyMath.min(delta_er, -1.1), 1.1)

            
            # print(f"delta_el: {delta_el} delta_er: {delta_er} w_l: {w_l} w_r: {w_r}")
            p, q, r = state[NUM_STATES-3:]

            va = state.v_a
            alpha_rad = state.alpha
            beta_rad = state.beta
            

            # Rotation matrix from Aerodynamic to Body frame
            R_BA = fi.Extract_R_BA(x)
            
            

            # Rotation matrix with Modified Rodrigues Parameters
            # sig_vec = np.array([sig1, sig2, sig3])
            # norm_sig_squared = np.dot(sig_vec, sig_vec)
            # sig_skew = np.array([[0, -sig3, sig2], [sig3, 0, -sig1], [-sig2, sig1, 0]])
            # R_BO = np.eye(3) + (8 * np.dot(sig_skew, sig_skew) - 4 * (1 - norm_sig_squared) * sig_skew) / (1 + norm_sig_squared)**2
            
            R_BO = fi.Extract_R_OB(state[:]).transpose()
            
            # Constants and parameters
            m = self.m  
            g = self.g 
            rho = self.rho 
            mu = self.mu
            nu = self.nu
            S = self.S
            S_w = 0.5*S
            c = self.c
            b = self.b
            l_motor = self.l_motor
            S_prop = self.S_prop
            k_motor = self.k_motor
            k_T_p = self.k_T_p
            k_omega = self.k_omega

            ## Aerodynamic coefficients for Zagi flying wing
            # Longitudinal aerodynamics
            C_L_0 = 0
            C_D_0 = 0
            C_m_0 = 0
            C_L_alpha = 3.5016
            C_D_alpha = .2108
            C_m_alpha = -.5675
            C_L_q = 2.8932
            C_D_q = 0
            C_m_q = -1.3990
            C_L_delta_e = .2724
            C_D_delta_e = .3045
            C_m_delta_e = -.3254
            C_prop = 1

            # Lateral aerodynamics
            C_Y_0 = 0
            C_l_0 = 0
            C_n_0 = 0
            C_Y_beta = -.07359
            C_l_beta = -.02854
            C_n_beta = -.00040
            C_Y_p = 0
            C_l_p = -.3209
            C_n_p = -.01297
            C_Y_r = 0
            C_l_r = .03066
            C_n_r = -.00434
            C_Y_delta_a = 0
            C_l_delta_a = .1682
            C_n_delta_a = -.00328
            
        

            # Forces due to aerodynamics
            coeffD = C_D_0 + C_D_alpha * pyMath.abs(alpha_rad)
            coeffM = C_m_0 + C_m_alpha * alpha_rad
            coeffL = coeffLf(alpha_rad)
            
            Fa_ra_x = rho * S / 2 * (coeffD * va**2 + (C_D_q * c * q * va) / 2 + C_D_delta_e * ( pyMath.abs(delta_el) + pyMath.abs(delta_er)) * va**2)
            Fa_ra_z = rho * S / 2 * (coeffL * va**2 + C_L_q * c * q * va / 2 + C_L_delta_e * (delta_el + delta_er) * va**2)
            Fa_ra_y = rho * S / 2 * (C_Y_0 * va**2 + C_Y_beta * beta_rad * va**2 + C_Y_p * b * r * va * p / 2 + C_Y_delta_a * (delta_el - delta_er) * va**2)

            # Propulsion forces
            Fthl_rb_x = rho * S_prop * C_prop * ((k_motor * w_l)**2 - va**2) / 2 
            Fthr_rb_x = rho * S_prop * C_prop * ((k_motor * w_r)**2 - va**2) / 2 
            F_thrust_tot = Fthl_rb_x + Fthr_rb_x

            # print(f"Fthl_rb_x: {Fthl_rb_x}")
            # print(f"F_thrust_tot: {F_thrust_tot}")
            # try:
                # Induced velocity by the propellers
            vi_squared = (va * pyMath.cos(alpha_rad))**2 + (2 * F_thrust_tot) / (rho * S_prop)
    
            vi = 0.5 * (np.sqrt(vi_squared) - va * pyMath.cos(alpha_rad)) #if vi_squared >= 0 else 0
            # vi = 0
            
            # print("vi_squared: ", vi_squared)
    
    
            # except Exception as e:
                # print("Error in vi: ", e)
                # vi = 0
                
            # vi = 0
            # print(f"vi: {vi}")
            # print(f"va: {va}")
            
            Fa_ra_x_vi = rho * S_w / 2 * C_D_delta_e * (pyMath.abs(delta_el) + pyMath.abs(delta_er)) * vi**2
            Fa_ra_z_vi = rho * S_w / 2 * C_L_delta_e * (delta_el + delta_er) * vi**2

            # Total aerodynamic forces
            F_aero_A_va = np.array([-Fa_ra_x, Fa_ra_y, -Fa_ra_z])
            F_aero_vi = np.array([-Fa_ra_x_vi, 0, -Fa_ra_z_vi])
            F = F_aero_vi + R_BA @ F_aero_A_va + np.array([Fthl_rb_x + Fthr_rb_x, 0, 0]) + R_BO @ np.array([0, 0, m * g])

            # print(f"f_aero_vi: {F_aero_vi}")
            # print(f"f_aero_A_va: {F_aero_A_va}")
            
            
            # Moments
            C_L_simplified_va = C_l_0 * va**2 + C_l_beta * beta_rad * va**2 + (C_l_p * b * p * va) / 2 + (C_l_r * b * r * va) / 2 + C_l_delta_a * (delta_el - delta_er) * va**2
            C_M_simplified_va = coeffM * va**2 + (C_m_q * c * q * va) / 2 + C_m_delta_e * (delta_el + delta_er) * va**2
            C_N_simplified_va = C_n_0 * va**2 + C_n_beta * beta_rad * va**2 + C_n_p * b * va / 2 * p + C_n_r * b * va / 2 * p + C_n_delta_a * (delta_el - delta_er) * va**2
            # changed one p to r 
            
            coeff_vector = np.array([C_L_simplified_va, C_M_simplified_va, C_N_simplified_va])
            tau_ab = 0.5 * rho * S * c * coeff_vector

            M_t_b = np.array([k_T_p * (k_omega * w_l)**2 - k_T_p * (k_omega * w_r)**2, 0, 0])
            M_p_b = np.array([0, 0, l_motor * (Fthl_rb_x - Fthr_rb_x)])

            C_M_vi = C_m_delta_e * (delta_el + delta_er) * vi**2
            M_prop = 0.5 * rho * S_w * c * np.array([0, C_M_vi, 0])
            M_elevon_lift = b / 4.0 * np.array([rho * S_w / 2 * (C_L_delta_e * (delta_el - delta_er) * vi**2), 0, 0])



            # print(f"differential delta : {(delta_el - delta_er)}")
            # print(f"Lift moment: {( (delta_el - delta_er) * vi**2)}")
            
            M = M_t_b + M_prop + M_p_b + tau_ab + M_elevon_lift

            # print F and M
            # print(f"M_t_b: {M_t_b} \nM_prop: {M_prop} \nM_p_b: {M_p_b} \ntau_ab: {tau_ab} \nM_elevon_lift: {M_elevon_lift}")
            # print(f"F: {F}")
            # print(f"M: {M}")
            
            return M, F


        def CopyStateOut(self, context, output):
            x = context.get_continuous_state_vector().CopyToVector()
            output.SetFromVector(x)
            # print("output updated")

        def OutputForces(self, context, output):
            M, F = self.ComputeForces(context)
            output.SetFromVector([M[0], M[1], M[2], F[0], F[1], F[2]])

    return Impl


FixedWingPlant = FixedWingPlant_[None]  # Default instantiation


# Add the camera_box.sdf.
def AddCameraBox(plant, X_WC, name="camera0", parent_frame=None):
    # TODO(russt): could be smarter and increment the default camera name (by
    # checking with the plant).
    if not parent_frame:
        parent_frame = plant.world_frame()
    parser = Parser(plant)
    ConfigureParser(parser)
    # directives = f"""
    #     directives:
    #     - add_model:
    #         name: {name}
    #         file: manipulation/models/camera_box.sdf
    #     """
    
    pwd = os.getcwd()
    camera_box_path = os.path.join(pwd, "models","camera_box.sdf")
    camera = parser.AddModels(camera_box_path)
    plant.WeldFrames(parent_frame, plant.GetFrameByName("base", camera[0]), X_WC)

    
# To use glider.urdf for visualization, follow the pattern from e.g.
# drake::examples::quadrotor::QuadrotorGeometry.

class FixedWingGeometry(LeafSystem):
    def __init__(self, scene_graph):
        LeafSystem.__init__(self)
        assert scene_graph

        mbp = MultibodyPlant(1.0)  # Time step doesn't matter, and this avoids a warning
        parser = Parser(mbp, scene_graph)
        parser.package_map().PopulateFromFolder("DeltaWingModel")
        
        deltawing_urdf = None
        try:
            PLANE_URDF_PATH = "DeltaWingModel/urdf/DeltaWing.urdf"
            with open(PLANE_URDF_PATH, 'r') as file:
                deltawing_urdf = file.read()
        except Exception as e:
            raise ValueError("Error reading the urdf file: ", e)
        
        

        
        (model_id, ) = parser.AddModelsFromString(deltawing_urdf, "urdf")

        # AddCameraBox(mbp, RigidTransform([-1, 0, -.3]), "camera0", parent_frame=mbp.GetFrameByName("base_link", model_id))
        mbp.Finalize()
        
        self.plant = mbp
        


        self.source_id = mbp.get_source_id()
        body_indices = mbp.GetBodyIndices(model_id)
        self.body_frame_id = mbp.GetBodyFrameIdOrThrow(body_indices[0])
        self.right_motor_frame_id = mbp.GetBodyFrameIdOrThrow(body_indices[1])
        self.left_motor_frame_id = mbp.GetBodyFrameIdOrThrow(body_indices[2])
        self.rigth_elevon_frame_id = mbp.GetBodyFrameIdOrThrow(body_indices[3])
        self.left_elevon_frame_id = mbp.GetBodyFrameIdOrThrow(body_indices[4])
        
        
        self.DeclareVectorInputPort("full_state", NUM_STATES+NUM_INPUTS)
        self.DeclareAbstractOutputPort(
            "geometry_pose",
            lambda: AbstractValue.Make(FramePoseVector()),
            self.OutputGeometryPose,
        )

    def OutputGeometryPose(self, context, poses):
        assert self.body_frame_id.is_valid()
        assert self.right_motor_frame_id.is_valid()
        assert self.left_motor_frame_id.is_valid()
        assert self.rigth_elevon_frame_id.is_valid()
        assert self.left_elevon_frame_id.is_valid()
        

        
        full_state = FullState(np.array(self.get_input_port(0).Eval(context),copy=True)) 
        # print(f"alpha: {full_state.alpha} beta: {full_state.beta}")# gamma: {full_state.gamma} mu: {full_state.mu} p: {full_state.p} q: {full_state.q} r: {full_state.r} delta_le: {full_state.delta_le} delta_re: {full_state.delta_re} delta_lm: {full_state.delta_lm} delta_rm: {full_state.delta_rm}")

        
        # print(f"PLOTTTING #########################################################################33")
        stateNED = fi.FixedWingStatesNED(np.array(full_state[:NUM_STATES], copy=True))
        
        X_DrB = fi.ExtractTransformation(stateNED)
        
        body_pose = X_DrB
        
        # From the urdf file
        R_Blm = RollPitchYaw(np.pi/2, 0, np.pi/2).ToRotationMatrix()
        p_Blm = [-0.013, -0.25, 0]
        X_Blm = RigidTransform(R_Blm, p_Blm)
        
        R_Brm = RollPitchYaw(np.pi/2, 0, np.pi/2).ToRotationMatrix()
        p_Brm = [-0.013011, 0.25, 0]
        X_Brm = RigidTransform(R_Brm, p_Brm)
        
        
        R_Bre = RollPitchYaw(0, 0, -0.17453).ToRotationMatrix()
        p_Bre = [-0.14351, 0.068516, 0]
        X_Bre = RigidTransform(R_Bre, p_Bre)
        
        R_Ble = RollPitchYaw(0, 0, 0.17453).ToRotationMatrix()
        p_Ble = [-0.14277, -0.073162, 0]    
        X_Ble = RigidTransform(R_Ble, p_Ble)
        
        
        # print(f"deltas: {full_state.delta_le} {full_state.delta_re} {full_state.delta_lm} {full_state.delta_rm}")
        
        
              
        scale = 40
        left_motor_pose = X_DrB @ X_Blm @ RigidTransform(RotationMatrix.MakeZRotation(-full_state.delta_lm* scale))
        right_motor_pose = X_DrB @ X_Brm @ RigidTransform(RotationMatrix.MakeZRotation(full_state.delta_rm* scale))
    
        left_elevon_pose = X_DrB @ X_Ble @ RigidTransform(RotationMatrix.MakeYRotation(full_state.delta_le ))
        right_elevon_pose = X_DrB @ X_Bre @ RigidTransform(RotationMatrix.MakeYRotation(full_state.delta_re ))
        

        poses.get_mutable_value().set_value(self.body_frame_id, body_pose)
        poses.get_mutable_value().set_value(self.right_motor_frame_id, right_motor_pose)
        poses.get_mutable_value().set_value(self.left_motor_frame_id, left_motor_pose)
        poses.get_mutable_value().set_value(self.rigth_elevon_frame_id, right_elevon_pose)
        poses.get_mutable_value().set_value(self.left_elevon_frame_id, left_elevon_pose)
        
        # print("body_pose: ", body_pose)
        
    @staticmethod
    def AddToBuilder(builder, fixedWing_state_port, scene_graph):
        assert builder
        assert scene_graph

        fw = FixedWingGeometry(scene_graph)
        geom = builder.AddSystem(fw)
        builder.Connect(fixedWing_state_port, geom.get_input_port(0))
        builder.Connect(
            geom.get_output_port(0),
            scene_graph.get_source_pose_port(geom.source_id),
        )
        
        # AddRgbdSensors(builder, fw.plant, scene_graph, 
        #                also_add_point_clouds=False, 
        #                model_instance_prefix="camera",
        #                depth_camera=None,
        #                renderer=None)

        return geom


def draw_glider(x, meshcat):
    builder = DiagramBuilder()
    glider = builder.AddSystem(FixedWingPlant())
    scene_graph = builder.AddSystem(SceneGraph())
    FixedWingGeometry.AddToBuilder(builder, glider.GetOutputPort("full_state"), scene_graph)
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    # meshcat.Set2dRenderMode(xmin=-4, xmax=1, ymin=-1, ymax=1)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    context.SetContinuousState(x)
    diagram.ForcedPublish(context)





def coeffLf(alpha_rad):
    alpha_deg = alpha_rad * 180 / np.pi
    va = 5
    c = 0.3302
    nu = 1.4776e-05
    Re = va * c / nu

    alpha0_deg = 9
    Re0 = 16e4
    pow_coef = 0.15
    c1_lift = 4.9781
    c2_lift = 1.015

    alpha_0_Re = alpha0_deg * (Re / Re0)**pow_coef
    

    # return .5
    try:
    
        if (pyMath.abs(alpha_deg) <= alpha_0_Re) or (pyMath.abs(alpha_deg) >= 180 - alpha_0_Re):
            CL1 = c1_lift * pyMath.sin(2 * pyMath.abs(alpha_rad))
            CL2 = 0
        elif (pyMath.abs(alpha_deg) > alpha_0_Re) and (pyMath.abs(alpha_deg) < 180 - alpha_0_Re):
            CL1 = 0
            CL2 = c2_lift * pyMath.sin(2 * pyMath.abs(alpha_rad))
        else:
            CL1 = 0
            CL2 = 0

        sigmoide = 1 / (1 + np.exp(pyMath.abs(alpha_deg) - alpha_0_Re)) + 1 / (1 + np.exp(180 - pyMath.abs(alpha_deg) - alpha_0_Re))
        coeffL = np.sign(alpha_deg) * (CL1 * sigmoide + CL2 * (1 - sigmoide))

        # print("coeffL: ", coeffL)
    except Exception as e:
        print("Error in coeffLf: ", e)
        coeffL = 0
    
    # print(f"coeffL: {coeffL}")
    return coeffL
