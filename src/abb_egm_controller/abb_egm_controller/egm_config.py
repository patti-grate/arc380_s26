from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from rcl_interfaces.msg import ParameterDescriptor
from rclpy.parameter import Parameter

DEFAULT_JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
DEFAULT_EXT_JOINT_NAMES = []
DEFAULT_EXT_JOINT_TYPES = []


class ControlSpace(Enum):
    JOINT = "joint"
    CARTESIAN = "cartesian"


@dataclass
class ControllerConfig:
    udp_port: int = 6510
    control_space: ControlSpace = ControlSpace.JOINT
    robot_joint_names: list[str] = field(default_factory=lambda: list(DEFAULT_JOINT_NAMES))
    ext_joint_names: list[str] = field(default_factory=lambda: list(DEFAULT_EXT_JOINT_NAMES))
    ext_joint_types: list[str] = field(default_factory=lambda: list(DEFAULT_EXT_JOINT_TYPES))
    docker_mode: bool = False
    relay_port_out: int = 6512

    def __setattr__(self, key, value):
        parsed_value = PARAM_SPECS[key].parser(value) if key in PARAM_SPECS else value
        super().__setattr__(key, parsed_value)

    def validate(self) -> None:
        if len(self.ext_joint_names) != len(self.ext_joint_types):
            raise ValueError(
                f"Mismatch between external joint names and types ({len(self.ext_joint_names)} names vs {len(self.ext_joint_types)} types)"
            )


@dataclass(frozen=True)
class ParamSpec:
    descriptor: ParameterDescriptor
    parser: Callable[[Parameter], Any]


# region Parsers


def _parse_udp_port(value: Any) -> int:
    if not isinstance(value, int):
        raise ValueError(f"UDP port must be an integer, got {type(value)}")
    if not (0 < value < 65536):
        raise ValueError(f"UDP port must be between 1 and 65535, got {value}")
    return value


def _parse_control_space(value: Any) -> ControlSpace:
    if not isinstance(value, (str, ControlSpace)):
        raise ValueError(f"Control space must be a string or ControlSpace enum, got {type(value)}")
    if isinstance(value, ControlSpace):
        return value
    try:
        return ControlSpace(value.lower())
    except ValueError:
        raise ValueError(f"Invalid control space: {value}")


def _parse_robot_joint_names(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"Robot joint names must be a list, got {type(value)}")
    names = [str(name) for name in value]
    if len(names) != 6:
        raise ValueError(f"Expected 6 robot joint names, got {len(names)}")
    # TODO: Check for empty names, duplicates, invalid chars, etc.
    return names


def _parse_ext_joint_names(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"External joint names must be a list, got {type(value)}")
    names = [str(name) for name in value]
    if len(names) > 6:
        raise ValueError(f"Expected at most 6 external joint names, got {len(names)}")
    # TODO: Check for empty names, duplicates, invalid chars, etc.
    return names


def _parse_ext_joint_types(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"External joint types must be a list, got {type(value)}")
    types = [str(type_).lower() for type_ in value]
    valid = {"prismatic", "revolute"}
    if len(types) > 6:
        raise ValueError(f"Expected at most 6 external joint types, got {len(types)}")
    for type_ in types:
        if type_ not in valid:
            raise ValueError(f"Invalid external joint type: {type_}. Allowed values are: {sorted(valid)}")
    return types


# endregion Parsers


PARAM_SPECS = {
    "udp_port": ParamSpec(
        descriptor=ParameterDescriptor(
            description="UDP port to listen for EGM messages",
            type=Parameter.Type.INTEGER,
        ),
        parser=_parse_udp_port,
    ),
    "control_space": ParamSpec(
        descriptor=ParameterDescriptor(
            description="Control space for streaming mode, either 'JOINT' or 'CARTESIAN'",
            type=Parameter.Type.STRING,
            additional_constraints="Allowed values: JOINT, CARTESIAN",
        ),
        parser=_parse_control_space,
    ),
    "robot_joint_names": ParamSpec(
        descriptor=ParameterDescriptor(
            description="Names of the robot joints in order",
            type=Parameter.Type.STRING_ARRAY,
        ),
        parser=_parse_robot_joint_names,
    ),
    "ext_joint_names": ParamSpec(
        descriptor=ParameterDescriptor(
            description="Names of the external joints in order",
            type=Parameter.Type.STRING_ARRAY,
        ),
        parser=_parse_ext_joint_names,
    ),
    "ext_joint_types": ParamSpec(
        descriptor=ParameterDescriptor(
            description="Types of the external joints in order: 'prismatic' or 'revolute'",
            type=Parameter.Type.STRING_ARRAY,
        ),
        parser=_parse_ext_joint_types,
    ),
    "docker_mode": ParamSpec(
        descriptor=ParameterDescriptor(
            description="Whether to operate in Docker relay mode (true) or direct control mode (false).",
            type=Parameter.Type.BOOL,
        ),
        parser=lambda value: bool(value),
    ),
    "relay_port_out": ParamSpec(
        descriptor=ParameterDescriptor(
            description="UDP port to send EGM messages to the relay (only used in Docker relay mode)",
            type=Parameter.Type.INTEGER,
        ),
        parser=_parse_udp_port,
    ),
}
