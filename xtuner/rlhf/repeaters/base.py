from ..policy_output import PolicyOutput


class RepeaterBase:
    """`RepeaterBase` is the base class of different repeaters.

    `repeater` is responsible to deal with the trajectory data.
    """

    def __init__(self):
        pass

    def process(self, trajectories: PolicyOutput, *args, **kwargs):
        """define process, such as get GAEs."""
        raise NotImplementedError
