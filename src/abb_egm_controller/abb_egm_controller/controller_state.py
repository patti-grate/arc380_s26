from __future__ import annotations

from enum import Enum

from abb_egm_interfaces.srv import SetControlMode

_ALLOWED_TRANSITIONS = [
    # to: F      I      S      T
    [True, True, False, False],  # from: F (FAULT)
    [True, True, True, True],  # from: I (IDLE)
    [True, True, True, True],  # from: S (STREAMING)
    [True, True, True, True],  # from: T (TRAJECTORY)
]

_REQUIRES_STOP = [
    # to: F      I      S      T
    [False, False, False, False],  # from: F (FAULT)
    [False, False, False, False],  # from: I (IDLE)
    [False, False, False, True],  # from: S (STREAMING)
    [False, False, True, False],  # from: T (TRAJECTORY)
]


class ControllerState(Enum):
    FAULT = 0
    IDLE = 1
    STREAMING = 2
    TRAJECTORY = 3

    @classmethod
    def from_srv(cls, value: int) -> ControllerState:
        for state in cls:
            if getattr(SetControlMode.Request, state.name) == value:
                return state
        raise ValueError(f"Invalid control state value: {value}")

    def to_srv(self) -> int:
        return getattr(SetControlMode.Request, self.name)

    def validate_transition(self, requested: ControllerState, stop_active_motion: bool) -> tuple[bool, str]:
        i = self.value
        j = requested.value
        if self == requested:
            return True, f"Already in {self.name} state"

        if not _ALLOWED_TRANSITIONS[i][j]:
            return False, f"Transition from {self.name} to {requested.name} is not allowed"

        if _REQUIRES_STOP[i][j] and not stop_active_motion:
            return False, f"Transition from {self.name} to {requested.name} requires stop_active_motion=True"

        return True, f"Transition from {self.name} to {requested.name} is valid"
