class Infer:
    def get_cmd(config=None):
        return "", config

    def validate(config=None):
        return True, "infer validate executed"
    
    def pre_action(config=None):
        return True, config

    def post_action(config=None):
        return True, config