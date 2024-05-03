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



import FlatnessInverter as fi

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
                context.get_mutable_continuous_state_vector().CopyToVector()
            )
            
            u = self.EvalVectorInput(context, 0).CopyToVector()
            s = fullstate[:NUM_STATES]
            
            F, M = self.forces_moments(s[:], u)
            
            return F, M

        def DoCalcTimeDerivatives(self, context, derivatives):
            fullstate = FullState(
                context.get_mutable_continuous_state_vector().CopyToVector()
            )
            
            u = self.EvalVectorInput(context, 0).CopyToVector()
            s = fi.FixedWingStatesNED(fullstate[:NUM_STATES])
            
            (delta_le, delta_re, w_lm, w_rm) = u
            
            F, M = self.forces_moments(s[:], u)
            X_a = F[0]
            Y_a = F[1]
            Z_a = F[2]
            L_b = M[0]
            M_b = M[1]
            N_b = M[2]
            
            
            fullstate_dot = FullState(fullstate[:])
            
            fullstate_dot.x = np.cos(s.chi) * np.cos(s.gamma) * s.v_a
            fullstate_dot.y = np.sin(s.chi) * np.cos(s.gamma) * s.v_a
            fullstate_dot.z = -np.sin(s.gamma) * s.v_a
            fullstate_dot.v_a = X_a/self.m - np.sin(s.gamma) * self.g
            
            fullstate_dot.beta = 0 # np.sin(s.alpha)*s.p - np.cos(s.alpha)*s.r + (np.cos(s.gamma)*np.sin(s.mu)*self.m*self.g + Y_a )/(self.m*s.v_a)
            fullstate_dot.alpha = 0 #s.q - (np.cos(s.alpha)*s.p + np.sin(s.alpha)*s.r)*np.tan(s.beta) +  (np.cos(s.gamma)*np.cos(s.mu)*self.g)/(np.cos(s.beta)*s.v_a) + Z_a/(np.cos(s.beta)*s.v_a*self.m)
            
            fullstate_dot.chi = (-Z_a * np.sin(s.mu) + Y_a * np.cos(s.mu)) / (s.v_a * self.m * np.cos(s.mu))
            fullstate_dot.gamma = (-np.cos(s.gamma) * self.g * self.m - Y_a * np.sin(s.mu) - Z_a * np.cos(s.mu)) / (s.v_a * self.m)
            
            mu_dot_num1 = -np.cos(s.mu) * np.cos(s.gamma) * np.sin(s.beta) * self.g
            mu_dot_num2 = s.p * np.cos(s.alpha) + s.r * np.sin(s.alpha)
            mu_dot_num3 = Z_a * np.sin(s.beta)
            mu_dot_num4 =np.sin(s.gamma) * (Y_a * np.cos(s.mu) - Z_a * np.sin(s.mu))

            # The corresponding denominators
            denom1 = s.v_a * np.cos(s.beta)
            denom2 = np.cos(s.beta)
            denom3 = s.v_a *self.m * np.cos(s.beta)
            denom4 = s.v_a * self.m * np.cos(s.gamma)
            

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
            lmbda = 1
            fullstate_dot.delta_le = -lmbda*fullstate.delta_le + delta_le 
            fullstate_dot.delta_re = -lmbda*fullstate.delta_re + delta_re
            fullstate_dot.delta_lm = w_lm
            fullstate_dot.delta_rm = w_rm
        
            
            derivatives.get_mutable_vector().SetFromVector(fullstate_dot[:])
            
            # print("derivatives updated")
            # print("fullstate: ", fullstate[:])
                    
        def forces_moments(self, x, u):
            # Unpack control inputs and state variables
            delta_el, delta_er, w_l, w_r = u
            state = fi.FixedWingStatesNED(x)
            
            p, q, r = state[NUM_STATES-3:]

            # Compute air data
            # va = np.sqrt(vb_x**2 + vb_y**2 + vb_z**2)
            # alpha_rad = np.arctan2(vb_z, vb_x)
            # alpha_deg = alpha_rad / np.pi * 180
            # beta_rad = np.arctan2(vb_y, va)
            # beta_deg = beta_rad / np.pi * 180
            
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
            coeffD = C_D_0 + C_D_alpha * np.abs(alpha_rad)
            coeffL = coeffLf(alpha_rad)
            Fa_ra_x = rho * S / 2 * (coeffD * va**2 + (C_D_q * c * q * va) / 2 + C_D_delta_e * (np.abs(delta_el) + np.abs(delta_er)) * va**2)
            Fa_ra_z = rho * S / 2 * (coeffL * va**2 + C_L_q * c * q * va / 2 + C_L_delta_e * (delta_el + delta_er) * va**2)
            Fa_ra_y = rho * S / 2 * (C_Y_0 * va**2 + C_Y_beta * beta_rad * va**2 + C_Y_p * b * r * va * p / 2 + C_Y_delta_a * (delta_el - delta_er) * va**2)

            # Propulsion forces
            Fthl_rb_x = rho * S_prop * C_prop * ((k_motor * w_l)**2 - va**2) / 2
            Fthr_rb_x = rho * S_prop * C_prop * ((k_motor * w_r)**2 - va**2) / 2
            F_thrust_tot = Fthl_rb_x + Fthr_rb_x

            try:
                # Induced velocity by the propellers
                vi_squared = (va * np.cos(alpha_rad))**2 + (2 * F_thrust_tot) / (rho * S_prop)
                
                vi = 0.5 * (np.sqrt(vi_squared) - va * np.cos(alpha_rad)) if vi_squared >= 0 else 0
                vi = 0
                
                # print("vi: ", vi)
            except Exception as e:
                print("Error in vi: ", e)
                vi = 0
            
            Fa_ra_x_vi = rho * S / 2 * C_D_delta_e * (np.abs(delta_el) + np.abs(delta_er)) * vi**2
            Fa_ra_z_vi = rho * S / 2 * C_L_delta_e * (delta_el + delta_er) * vi**2

            # Total aerodynamic forces
            F_aero_A_va = np.array([-Fa_ra_x, Fa_ra_y, -Fa_ra_z])
            F_aero_vi = np.array([-Fa_ra_x_vi, 0, -Fa_ra_z_vi])
            F = F_aero_vi + R_BA @ F_aero_A_va + np.array([Fthl_rb_x + Fthr_rb_x, 0, 0]) + R_BO @ np.array([0, 0, m * g])

            # Moments
            C_L_simplified_va = C_l_0 * va**2 + C_l_beta * beta_rad * va**2 + (C_l_p * b * p * va) / 2 + (C_l_r * b * r * va) / 2 + C_l_delta_a * (delta_el - delta_er) * va**2
            C_M_simplified_va = coeffL * va**2 + (C_m_q * c * q * va) / 2 + C_m_delta_e * (delta_el + delta_er) * va**2
            C_N_simplified_va = C_n_0 * va**2 + C_n_beta * beta_rad * va**2 + C_n_p * b * va / 2 * p + C_n_r * b * va / 2 * r + C_n_delta_a * (delta_el - delta_er) * va**2

            coeff_vector = np.array([C_L_simplified_va, C_M_simplified_va, C_N_simplified_va])
            tau_ab = 0.5 * rho * S * c * coeff_vector

            M_t_b = np.array([k_T_p * (k_omega * w_l)**2 - k_T_p * (k_omega * w_r)**2, 0, 0])
            M_p_b = np.array([0, 0, l_motor * (Fthl_rb_x - Fthr_rb_x)])

            C_M_vi = C_m_delta_e * (delta_el + delta_er) * vi**2
            M_prop = 0.5 * rho * S * c * np.array([0, C_M_vi, 0])
            M_elevon_lift = b / 4 * np.array([rho * S / 2 * (C_L_delta_e * (delta_el - delta_er) * vi**2), 0, 0])

            M = M_t_b + M_prop + M_p_b + tau_ab + M_elevon_lift

            return F, M, 


        def CopyStateOut(self, context, output):
            x = context.get_continuous_state_vector().CopyToVector()
            output.SetFromVector(x)
            # print("output updated")

        def OutputForces(self, context, output):
            # F, M = self.ComputeForces(context)
            F = np.zeros(3)
            M = np.zeros(3)
            output.SetFromVector([M[0], M[1], M[2], F[0], F[1], F[2]])

    return Impl


FixedWingPlant = FixedWingPlant_[None]  # Default instantiation


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

        mbp.Finalize()


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
        

        
        full_state = FullState(self.get_input_port(0).Eval(context)) 
        # print(f"alpha: {full_state.alpha} beta: {full_state.beta}")# gamma: {full_state.gamma} mu: {full_state.mu} p: {full_state.p} q: {full_state.q} r: {full_state.r} delta_le: {full_state.delta_le} delta_re: {full_state.delta_re} delta_lm: {full_state.delta_lm} delta_rm: {full_state.delta_rm}")

        
        stateNED = fi.FixedWingStatesNED(full_state[:NUM_STATES])
        
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
        
              
        left_motor_pose = X_DrB @ X_Blm @ RigidTransform(RotationMatrix.MakeZRotation(-full_state.delta_lm))
        right_motor_pose = X_DrB @ X_Brm @ RigidTransform(RotationMatrix.MakeZRotation(full_state.delta_rm))
        
        left_elevon_pose = X_DrB @ X_Ble @ RigidTransform(RotationMatrix.MakeYRotation(full_state.delta_le))
        right_elevon_pose = X_DrB @ X_Bre @ RigidTransform(RotationMatrix.MakeYRotation(full_state.delta_re))
        

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

        geom = builder.AddSystem(FixedWingGeometry(scene_graph))
        builder.Connect(fixedWing_state_port, geom.get_input_port(0))
        builder.Connect(
            geom.get_output_port(0),
            scene_graph.get_source_pose_port(geom.source_id),
        )

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
    
        if (np.abs(alpha_deg) <= alpha_0_Re) or (np.abs(alpha_deg) >= 180 - alpha_0_Re):
            CL1 = c1_lift * np.sin(2 * np.abs(alpha_rad))
            CL2 = 0
        elif (np.abs(alpha_deg) > alpha_0_Re) and (np.abs(alpha_deg) < 180 - alpha_0_Re):
            CL1 = 0
            CL2 = c2_lift * np.sin(2 * np.abs(alpha_rad))
        else:
            CL1 = 0
            CL2 = 0

        sigmoide = 1 / (1 + np.exp(np.abs(alpha_deg) - alpha_0_Re)) + 1 / (1 + np.exp(180 - np.abs(alpha_deg) - alpha_0_Re))
        coeffL = np.sign(alpha_deg) * (CL1 * sigmoide + CL2 * (1 - sigmoide))

        # print("coeffL: ", coeffL)
    except Exception as e:
        print("Error in coeffLf: ", e)
        coeffL = 0
    
    return coeffL

# def Ry(theta):
#     return np.array([[np.cos(theta), 0, np.sin(theta)],
#                      [0, 1, 0],
#                      [-np.sin(theta), 0, np.cos(theta)]])

# def Rz(theta):
#     return np.array([[np.cos(theta), -np.sin(theta), 0],
#                      [np.sin(theta), np.cos(theta), 0],
#                      [0, 0, 1]])
