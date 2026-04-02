from math import degrees, radians

from abb_egm_interfaces.msg import EgmRobot, EgmSensor
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

from abb_egm_controller import egm_pb2 as egm

DEFAULT_JOINT_NAMES = [f"joint_{i + 1}" for i in range(6)]
DEFAULT_EXT_JOINT_NAMES = [f"ext_{i + 1}" for i in range(6)]
DEFAULT_JOINT_TYPES = ["revolute"] * 6


def egm_header_to_ros(egm_header) -> Header:
    if not egm_header.HasField("tm"):
        return Header()

    time_ms = egm_header.tm

    header = Header()
    header.stamp.sec = int(time_ms // 1000)
    header.stamp.nanosec = int((time_ms % 1000) * 1_000_000)
    return header


def ros_header_to_egm(msg: Header):
    time_ms = msg.stamp.sec * 1000 + msg.stamp.nanosec // 1_000_000

    egm_header = egm.EgmHeader()
    egm_header.tm = int(time_ms)
    return egm_header


def egm_clock_to_ros_header(egm_clock) -> Header:
    header = Header()
    header.stamp.sec = egm_clock.sec
    header.stamp.nanosec = int(egm_clock.usec * 1000)
    return header


def ros_header_to_egm_clock(header: Header):
    egm_clock = egm.EgmClock()
    egm_clock.sec = header.stamp.sec
    egm_clock.usec = int(header.stamp.nanosec // 1000)
    return egm_clock


def egm_pose_to_ros(egm_pose) -> Pose:
    pose = Pose()
    # Convert from mm (EGM) to m (ROS)
    pose.position.x = egm_pose.pos.x / 1000.0
    pose.position.y = egm_pose.pos.y / 1000.0
    pose.position.z = egm_pose.pos.z / 1000.0
    pose.orientation.w = egm_pose.orient.u0
    pose.orientation.x = egm_pose.orient.u1
    pose.orientation.y = egm_pose.orient.u2
    pose.orientation.z = egm_pose.orient.u3
    return pose


def ros_pose_to_egm(pose: Pose):
    egm_pose = egm.EgmPose()
    # Convert from m (ROS) to mm (EGM)
    egm_pose.pos.x = pose.position.x * 1000.0
    egm_pose.pos.y = pose.position.y * 1000.0
    egm_pose.pos.z = pose.position.z * 1000.0
    egm_pose.orient.u0 = pose.orientation.w
    egm_pose.orient.u1 = pose.orientation.x
    egm_pose.orient.u2 = pose.orientation.y
    egm_pose.orient.u3 = pose.orientation.z
    return egm_pose


def egm_joints_to_ros(egm_joints, *, joint_names: list[str] | None = None) -> JointState:
    joint_names = DEFAULT_JOINT_NAMES.copy() if joint_names is None else joint_names
    if len(joint_names) != 6:
        raise ValueError(f"Expected 6 joint names, got {len(joint_names)}")
    joint_state = JointState()
    joint_state.name = joint_names
    joint_state.position = [radians(val) for val in list(egm_joints.joints)]
    return joint_state


def ros_joints_to_egm(joint_state: JointState, *, joint_types: list[str] | None = None):
    joint_types = DEFAULT_JOINT_TYPES.copy() if joint_types is None else joint_types
    if len(joint_types) != 6:
        raise ValueError(f"Expected 6 joint types, got {len(joint_types)}")
    joint_vals = []
    for val, joint_type in zip(joint_state.position, joint_types):
        if joint_type == "revolute":
            joint_vals.append(degrees(val))  # Convert from rad (ROS) to deg (EGM)
        elif joint_type == "prismatic":
            joint_vals.append(val * 1000.0)  # Convert from m (ROS) to mm (EGM)
        else:
            raise ValueError(f"Unsupported joint type: {joint_type}")
    egm_joints = egm.EgmJoints()
    egm_joints.joints.extend(joint_vals)
    return egm_joints


def egm_ext_joints_to_ros(
    egm_ext_joints, *, joint_names: list[str] | None = None, joint_types: list[str] | None = None
) -> JointState:
    joint_names = DEFAULT_EXT_JOINT_NAMES.copy() if joint_names is None else joint_names
    joint_types = DEFAULT_JOINT_TYPES.copy() if joint_types is None else joint_types
    if len(joint_names) != 6:
        raise ValueError(
            f"Expected 6 external joint names, got {len(joint_names)}"
        )  # TODO: Check if 6 is always expected for external joints
    if len(joint_types) != 6:
        raise ValueError(
            f"Expected 6 external joint types, got {len(joint_types)}"
        )  # TODO: Check if 6 is always expected for external joints
    joint_vals = []
    for val, joint_type in zip(egm_ext_joints.joints, joint_types):
        if joint_type == "revolute":
            joint_vals.append(radians(val))  # Convert from deg (EGM) to rad (ROS)
        elif joint_type == "prismatic":
            joint_vals.append(val / 1000.0)  # Convert from mm (EGM) to m (ROS)
        else:
            raise ValueError(f"Unsupported joint type: {joint_type}")
    joint_state = JointState()
    joint_state.name = joint_names
    joint_state.position = joint_vals
    return joint_state


def egm_robot_to_ros(
    egm_robot,
    *,
    robot_joint_names: list[str] | None = None,
    ext_joint_names: list[str] | None = None,
    ext_joint_types: list[str] | None = None,
) -> EgmRobot:
    msg = EgmRobot()

    if egm_robot.HasField("header"):
        msg.header = egm_header_to_ros(egm_robot.header)
        msg.msg_type = int(egm_robot.header.mtype)

    if egm_robot.HasField("feedBack"):
        if egm_robot.feedBack.HasField("time"):
            feedback_header = egm_clock_to_ros_header(egm_robot.feedBack.time)
        else:
            feedback_header = Header()

        if egm_robot.feedBack.HasField("joints"):
            msg.feedback_joints = egm_joints_to_ros(egm_robot.feedBack.joints, joint_names=robot_joint_names)
            msg.feedback_joints.header = feedback_header
        if egm_robot.feedBack.HasField("cartesian"):
            msg.feedback_pose.pose = egm_pose_to_ros(egm_robot.feedBack.cartesian)
            msg.feedback_pose.header = feedback_header
        if egm_robot.feedBack.HasField("externalJoints"):
            msg.feedback_ext_joints = egm_ext_joints_to_ros(
                egm_robot.feedBack.externalJoints, joint_names=ext_joint_names, joint_types=ext_joint_types
            )
            msg.feedback_ext_joints.header = feedback_header

    if egm_robot.HasField("planned"):
        if egm_robot.planned.HasField("time"):
            planned_header = egm_clock_to_ros_header(egm_robot.planned.time)
        else:
            planned_header = Header()

        if egm_robot.planned.HasField("joints"):
            msg.planned_joints = egm_joints_to_ros(egm_robot.planned.joints, joint_names=robot_joint_names)
            msg.planned_joints.header = planned_header
        if egm_robot.planned.HasField("cartesian"):
            msg.planned_pose.pose = egm_pose_to_ros(egm_robot.planned.cartesian)
            msg.planned_pose.header = planned_header
        if egm_robot.planned.HasField("externalJoints"):
            msg.planned_ext_joints = egm_ext_joints_to_ros(
                egm_robot.planned.externalJoints, joint_names=ext_joint_names, joint_types=ext_joint_types
            )
            msg.planned_ext_joints.header = planned_header

    if egm_robot.HasField("motorState"):
        msg.motor_state = int(egm_robot.motorState.state)
    if egm_robot.HasField("mciState"):
        msg.mci_state = int(egm_robot.mciState.state)
    if egm_robot.HasField("mciConvergenceMet"):
        msg.mci_convergence_met = bool(egm_robot.mciConvergenceMet)

    if egm_robot.HasField("testSignals"):
        msg.test_signals = list(egm_robot.testSignals.signals)

    if egm_robot.HasField("rapidExecState"):
        msg.rapid_exec_state = int(egm_robot.rapidExecState.state)

    if egm_robot.HasField("measuredForce"):
        if egm_robot.measuredForce.HasField("fcActive"):
            msg.measured_force_active = bool(egm_robot.measuredForce.fcActive)
        msg.measured_force = list(egm_robot.measuredForce.force)

    if egm_robot.HasField("utilizationRate"):
        msg.utilization_rate = egm_robot.utilizationRate
    if egm_robot.HasField("moveIndex"):
        msg.move_index = egm_robot.moveIndex

    if egm_robot.HasField("CollisionInfo"):
        if egm_robot.CollisionInfo.HasField("collsionTriggered"):
            msg.collision_triggered = bool(egm_robot.CollisionInfo.collsionTriggered)
        msg.collision_quota = list(egm_robot.CollisionInfo.collDetQuota)

    if egm_robot.HasField("RAPIDfromRobot"):
        if egm_robot.RAPIDfromRobot.HasField("digVal"):
            msg.rapid_dig_val = bool(egm_robot.RAPIDfromRobot.digVal)
        msg.rapid_dnum = list(egm_robot.RAPIDfromRobot.dnum)

    return msg


def ros_sensor_to_egm(sensor_msg: EgmSensor, *, ext_joint_types: list[str] | None = None):
    egm_sensor = egm.EgmSensor()

    egm_sensor.header.CopyFrom(ros_header_to_egm(sensor_msg.header))
    egm_sensor.header.mtype = int(sensor_msg.msg_type)

    planned_headers = []
    if sensor_msg.mode == EgmSensor.MODE_JOINTS:
        if len(sensor_msg.planned_joints.position) < 6:
            raise ValueError("Expected at least 6 joint positions for MODE_JOINTS")
        egm_sensor.planned.joints.CopyFrom(ros_joints_to_egm(sensor_msg.planned_joints))
        planned_headers.append(sensor_msg.planned_joints.header)
    if sensor_msg.mode == EgmSensor.MODE_CARTESIAN:
        egm_sensor.planned.cartesian.CopyFrom(ros_pose_to_egm(sensor_msg.planned_pose.pose))
        planned_headers.append(sensor_msg.planned_pose.header)
    if len(sensor_msg.planned_ext_joints.position) > 0:
        egm_sensor.planned.externalJoints.CopyFrom(
            ros_joints_to_egm(sensor_msg.planned_ext_joints, joint_types=ext_joint_types)
        )
        planned_headers.append(sensor_msg.planned_ext_joints.header)

    latest_header = max(planned_headers, key=lambda h: (h.stamp.sec, h.stamp.nanosec))
    if latest_header.stamp.sec != 0 or latest_header.stamp.nanosec != 0:
        egm_sensor.planned.time.CopyFrom(ros_header_to_egm_clock(latest_header))

    if sensor_msg.speed_ref_joints:
        if len(sensor_msg.speed_ref_joints) != 6:
            raise ValueError("Expected 6 joint speed references if speed_ref_joints is provided")
        speed_ref_joints = [degrees(val) for val in list(sensor_msg.speed_ref_joints)]
        egm_sensor.speedRef.joints.joints.extend(speed_ref_joints)
    if sensor_msg.speed_ref_cartesian:
        if len(sensor_msg.speed_ref_cartesian) != 6:
            raise ValueError(
                "Expected 6 Cartesian speed references (x, y, z, r, p, y) if speed_ref_cartesian is provided"
            )
        speed_ref_cartesian = [val * 1000.0 for val in list(sensor_msg.speed_ref_cartesian)]
        egm_sensor.speedRef.cartesians.value.extend(speed_ref_cartesian)
    if sensor_msg.speed_ref_ext_joints:
        speed_ref_ext_joints = []
        for val, joint_type in zip(sensor_msg.speed_ref_ext_joints, ext_joint_types):
            if joint_type == "revolute":
                speed_ref_ext_joints.append(degrees(val))  # Convert from rad (ROS) to deg (EGM)
            elif joint_type == "prismatic":
                speed_ref_ext_joints.append(val * 1000.0)  # Convert from m/s (ROS) to mm/s (EGM)
            else:
                raise ValueError(f"Unsupported joint type: {joint_type}")
        egm_sensor.speedRef.externalJoints.joints.extend(speed_ref_ext_joints)

    if sensor_msg.send_rapid_data:
        # TODO: Configure once \DIFromSensor is working in RAPID
        # egm_sensor.RAPIDtoRobot.digVal = bool(sensor_msg.rapid_dig_val)
        if sensor_msg.rapid_dnum:
            egm_sensor.RAPIDtoRobot.dnum.extend(sensor_msg.rapid_dnum)
    return egm_sensor
