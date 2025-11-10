class Eval:
    def get_cmd(config=None):
        return "echo 'eval success'"

    def validate(config=None):
        return True, "eval validate executed"
