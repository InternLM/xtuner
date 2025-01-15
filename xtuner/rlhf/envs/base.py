class EnvBase:
    """`EnvBase` is the base class of different environments.

    `env` is responsible to generate the trajectory data.
    """

    def __init__(self):
        pass

    def rollout(self, *args, **kwargs):
        """define rollout."""
        raise NotImplementedError
