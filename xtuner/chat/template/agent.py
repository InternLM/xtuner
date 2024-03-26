from dataclasses import dataclass

from .chat import ChatTemplateConfig


@dataclass
class AgentTemplateConfig(ChatTemplateConfig):
    system_with_meta: str
    environment: str
    environment_with_meta: str
    instruction_with_file: str
    interpreter_token: str
    plugin_token: str
    action_start_token: str
    action_end_token: str

    def template_map_fn_v2(self, example):
        messages_original = example['messages']
        messages = []
        n_turn = 1
        for data_idx, data in enumerate(messages_original):
            role_original = data.get('role')
            content_original = data.get('content')
            extra = data.get('extra')
            if role_original == 'system':
                role = 'system'
                if extra is None:
                    content = self.system.format(system=content_original)
                else:
                    meta_type = extra.get('meta_type')
                    if meta_type == 'interpreter':
                        content = self.system_with_meta.format(
                            system=content_original,
                            meta=self.interpreter_token)
                    elif meta_type == 'plugin':
                        content = self.system_with_meta.format(
                            system=content_original, meta=self.plugin_token)
                    else:
                        raise NotImplementedError
            elif role_original == 'environment':
                role = 'system'
                if extra is None:
                    content = self.environment.format(
                        environment=content_original)
                else:
                    meta_type = extra.get('meta_type')
                    if meta_type == 'interpreter':
                        content = self.environment_with_meta.format(
                            environment=content_original,
                            meta=self.interpreter_token)
                    elif meta_type == 'plugin':
                        content = self.environment_with_meta.format(
                            environment=content_original,
                            meta=self.plugin_token)
                    else:
                        raise NotImplementedError
            elif role_original == 'user':
                role = 'user'
                if extra is None:
                    content = self.instruction.format(
                        input=content_original, round=n_turn)
                else:
                    upload_type = extra.get('upload_type')
                    upload_content = extra.get('upload_content')
                    assert upload_content is not None
                    if upload_type == 'file':
                        content = self.instruction_with_file.format(
                            input=content_original,
                            file=upload_content,
                            round=n_turn)
                    else:
                        raise NotImplementedError
                n_turn += 1
            elif role_original == 'assistant':
                role = 'assistant'
                if extra is None:
                    content = content_original
                else:
                    action_type = extra.get('action_type')
                    action_content = extra.get('action_content')
                    assert action_content is not None
                    content = content_original + '\n' + self.action_start_token
                    if action_type == 'interpreter':
                        content += self.interpreter_token + '\n'
                    elif action_type == 'plugin':
                        content += self.plugin_token + '\n'
                    else:
                        raise NotImplementedError
                    content += action_content + self.action_end_token + '\n'
                if self.suffix != '':
                    content += self.suffix
            else:
                raise NotImplementedError
            messages.append({'role': role, 'content': content})
        return {'messages': messages}
