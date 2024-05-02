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

from FlatnessInverter import FixedWingStatesNED

import FlatnessInverter as fi

NUM_STATES = 12
NUM_INPUTS = 4



FullState = namedview(
    "FullState",
    [
        "x", "y", "z", "v_a", "betha", "alpha", "chi", "gamma", "mu", "p", "q", "r",
        "delta_le", "delta_re", "delta_lm", "delta_rm"
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
            self.DeclareVectorOutputPort("forces", 2, self.OutputForces)

            # parameters based on Rick Cory's "R1 = no dihedral" model.
            self.Sw = 0.0885  # surface area of wing + fuselage + tail.
            self.Se = 0.0147  # surface area of elevator.
            self.lw = 0  # horizontal offset of wing center.
            self.le = 0.022  # elevator aerodynamic center from hinge.
            # self.lh = 0.317  # elevator hinge.
            self.lh = 0.27  # elevator hinge.
            self.inertia = 0.0015  # body inertia.
            self.m = 0.08  # body mass.
            self.rho = 1.204  # air density (kg/m^3).
            self.gravity = 9.81  # gravity

            # TODO(russt): Declare elevator constraints:
            self.elevator_lower_limit = -0.9473
            self.elevator_upper_limit = 0.4463

        def _construct_copy(self, other, converter=None):
            Impl._construct(self, converter=converter)

        def ComputeForces(self, context):
            # s = FixedWingStatesNED(
            #     context.get_mutable_continuous_state_vector().CopyToVector()
            # )
            # print(self.EvalVectorInput(context, 0))
            (delta_le, delta_re, w_lm, w_rm)  = self.EvalVectorInput(context, 0).CopyToVector()

            # eps = 1e-10
            # xwdot = s.xdot + self.lw * s.pitchdot * np.sin(s.pitch)
            # zwdot = s.zdot + self.lw * s.pitchdot * np.cos(s.pitch)
            # vw = np.sqrt(zwdot**2 + xwdot**2 + eps)
            # fw = (
            #     -self.rho
            #     * self.Sw
            #     * (np.sin(s.pitch) * xwdot + np.cos(s.pitch) * zwdot)
            #     * vw
            # )

            # e = s.pitch + s.elevator
            # edot = s.pitchdot + elevatordot
            # xedot = (
            #     s.xdot
            #     + self.lh * s.pitchdot * np.sin(s.pitch)
            #     + self.le * edot * np.sin(e)
            # )
            # zedot = (
            #     s.zdot
            #     + self.lh * s.pitchdot * np.cos(s.pitch)
            #     + self.le * edot * np.cos(e)
            # )
            # ve = np.sqrt(zedot**2 + xedot**2 + eps)
            # fe = -self.rho * self.Se * (np.sin(e) * xedot + np.cos(e) * zedot) * ve

            fw , fe = 0, 0
            return fw, fe

        def DoCalcTimeDerivatives(self, context, derivatives):
            # s = FixedWingStatesNED(
            #     context.get_mutable_continuous_state_vector().CopyToVector()
            # )
            # elevatordot = self.EvalVectorInput(context, 0)[0]
            # e = s.pitch + s.elevator
            # fw, fe = self.ComputeForces(context)

            # sdot = FixedWingStatesNED(s[:])
            # sdot[0:3] = s[4:7]
            # sdot.elevator = elevatordot
            # sdot.xdot = (fw * np.sin(s.pitch) + fe * np.sin(e)) / self.m
            # sdot.zdot = (fw * np.cos(s.pitch) + fe * np.cos(e)) / self.m - self.gravity
            # sdot.pitchdot = (
            #     fw * self.lw + fe * (self.lh * np.cos(s.elevator) + self.le)
            # ) / self.inertia
            sdot = np.zeros(NUM_STATES+NUM_INPUTS)
            derivatives.get_mutable_vector().SetFromVector(sdot)

        def CopyStateOut(self, context, output):
            x = context.get_continuous_state_vector().CopyToVector()
            output.SetFromVector(x)

        def OutputForces(self, context, output):
            fw, fe = self.ComputeForces(context)
            output.SetFromVector([fw, fe])

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
        
        stateNED = FixedWingStatesNED(full_state[:NUM_STATES])
        
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
    FixedWingGeometry.AddToBuilder(builder, glider.GetOutputPort("state"), scene_graph)
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    # meshcat.Set2dRenderMode(xmin=-4, xmax=1, ymin=-1, ymax=1)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    context.SetContinuousState(x)
    diagram.ForcedPublish(context)


def dircol_perching():
    builder = DiagramBuilder()
    glider = builder.AddSystem(FixedWingPlant())
    # scene_graph = builder.AddSystem(SceneGraph())
    # FixedWingGeometry.AddToBuilder(builder, glider.GetOutputPort("state"), scene_graph)
    # visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    # meshcat.Delete()
    # meshcat.Set2dRenderMode(xmin=-4, xmax=1, ymin=-1, ymax=1)

    # diagram = builder.Build()
    # context = diagram.CreateDefaultContext()
    # diagram.ForcedPublish(context)

    x_traj = None
    u_traj = None

    # SNOPT is more reliable if we solve it twice.
    for N in [25, 41]:
        dircol = DirectCollocation(
            glider, glider.CreateDefaultContext(), N, 0.5 / N, 2.0 / N
        )
        prog = dircol.prog()
        dircol.AddEqualTimeIntervalsConstraints()

        # Input limits
        u = dircol.input()
        elevator_velocity_limit = 13  # max servo velocity (rad/sec)
        dircol.AddConstraintToAllKnotPoints(-elevator_velocity_limit <= u[0])
        dircol.AddConstraintToAllKnotPoints(u[0] <= elevator_velocity_limit)

        # Initial conditions
        s0 = FixedWingStatesNED(np.zeros(7))
        s0.x = -3.5
        # s0.z = 0.1
        s0.z = 1.1  
        s0.xdot = 7.0
        prog.AddBoundingBoxConstraint(s0[:], s0[:], dircol.initial_state())
        # context.SetContinuousState(s0[:])
        # diagram.ForcedPublish(context)

        # Final conditions
        sf = FixedWingStatesNED(dircol.final_state())
        prog.AddBoundingBoxConstraint(0, 0, sf.x)
        # prog.AddBoundingBoxConstraint(0, 0, sf.z)
        prog.AddBoundingBoxConstraint(1, 1, sf.z)
        prog.AddBoundingBoxConstraint(-1.0, -np.pi / 6.0, sf.pitch)
        prog.AddBoundingBoxConstraint(-2.0, 2.0, sf.xdot)
        prog.AddBoundingBoxConstraint(-2.0, 2.0, sf.zdot)

        # State constraints
        s = FixedWingStatesNED(dircol.state())
        dircol.AddConstraintToAllKnotPoints(
            s.x >= s0.x
        )  # TODO: s0.x <= s.x gives an error.
        dircol.AddConstraintToAllKnotPoints(s.x <= 1)
        # dircol.AddConstraintToAllKnotPoints(s.z >= -1)
        # dircol.AddConstraintToAllKnotPoints(s.z <= 1)
        dircol.AddConstraintToAllKnotPoints(s.z >= 0)
        dircol.AddConstraintToAllKnotPoints(s.z <= 2)

        dircol.AddConstraintToAllKnotPoints(s.pitch >= -np.pi / 2.0)
        dircol.AddConstraintToAllKnotPoints(s.pitch <= np.pi / 2.0)
        dircol.AddConstraintToAllKnotPoints(s.elevator >= glider.elevator_lower_limit)
        dircol.AddConstraintToAllKnotPoints(s.elevator <= glider.elevator_upper_limit)

        # Cost
        dircol.AddRunningCost(100 * u**2)

        sf_d = FixedWingStatesNED(np.zeros(7))
        sf_d.pitch = -np.pi / 4.0
        # sf_d.z= 0
        sf_d.z= 1
        
        prog.AddQuadraticErrorCost(
            np.diag([10, 10, 1, 10, 1, 1, 1]), sf_d[:], dircol.final_state()
        )

        # def plot_trajectory(times, states):
        #     s = FixedWingStatesNED(states)
        #     vertices = np.vstack([s.x, 0 * s.x, s.z])
        #     meshcat.SetLine("dircol", vertices, rgba=Rgba(0, 0, 0.5))

        # dircol.AddStateTrajectoryCallback(plot_trajectory)

        if x_traj and u_traj:
            dircol.SetInitialTrajectory(u_traj, x_traj)
        else:
            initial_x_trajectory = PiecewisePolynomial.FirstOrderHold(
                [0.0, 0.8], np.column_stack((s0[:], sf_d[:]))
            )
            dircol.SetInitialTrajectory(PiecewisePolynomial(), initial_x_trajectory)
        result = Solve(prog)
        assert result.is_success()

        x_traj = dircol.ReconstructStateTrajectory(result)
        u_traj = dircol.ReconstructInputTrajectory(result)

    # # Animate trajectory
    # visualizer.StartRecording()
    # for t in np.hstack(
    #     (
    #         np.arange(x_traj.start_time(), x_traj.end_time(), 1.0 / 32.0),
    #         x_traj.end_time(),
    #     )
    # ):
    #     context.SetTime(t)
    #     context.SetContinuousState(x_traj.value(t))
    #     diagram.ForcedPublish(context)

    # visualizer.StopRecording()
    # visualizer.PublishRecording()

    return x_traj, u_traj