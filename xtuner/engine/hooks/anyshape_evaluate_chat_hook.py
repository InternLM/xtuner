# Copyright (c) OpenMMLab. All rights reserved.
import torch

from xtuner.dataset.utils import process_anyres_image
from xtuner.model.utils import prepare_inputs_labels_for_multimodal
from xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from .evaluate_chat_hook import EvaluateChatHook


class AnyShapeEvaluateChatHook(EvaluateChatHook):

    def __init__(self, image_grid_pinpoints, *args, **kwargs):
        self.image_grid_pinpoints = image_grid_pinpoints
        super().__init__(*args, **kwargs)

    def _eval_images(self,
                     runner,
                     model,
                     device,
                     max_new_tokens=None,
                     save_eval_output=False):
        if save_eval_output:
            eval_outputs = []

        for sample_image, sample_input in zip(self.evaluation_images,
                                              self.evaluation_inputs):

            orig_size = sample_image.size
            # n,c,h,w
            image = process_anyres_image(sample_image, self.image_processor,
                                         self.image_grid_pinpoints)

            image = image.to(device)
            sample_input = DEFAULT_IMAGE_TOKEN + '\n' + sample_input
            inputs = (self.system + self.instruction).format(
                input=sample_input, round=1, **runner.cfg)
            chunk_encode = []
            for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
                if idx == 0:
                    cur_encode = self.tokenizer.encode(chunk)
                else:
                    cur_encode = self.tokenizer.encode(
                        chunk, add_special_tokens=False)
                chunk_encode.append(cur_encode)
            assert len(chunk_encode) == 2
            input_ids = []
            for idx, cur_chunk_encode in enumerate(chunk_encode):
                input_ids.extend(cur_chunk_encode)
                if idx != len(chunk_encode) - 1:
                    input_ids.append(IMAGE_TOKEN_INDEX)
            input_ids = torch.tensor(input_ids).to(device)

            image_features = model.preprocess_for_pixel_values({
                'pixel_values':
                image.unsqueeze(0),
                'orig_sizes': [orig_size]
            })

            mm_inputs = prepare_inputs_labels_for_multimodal(
                llm=model.llm,
                input_ids=input_ids.unsqueeze(0),
                pixel_values=image_features)

            generation_output = model.generate(
                **mm_inputs,
                max_new_tokens=max_new_tokens,
                generation_config=self.gen_config,
                bos_token_id=self.tokenizer.bos_token_id,
                stopping_criteria=self.stop_criteria)
            generation_output = self.tokenizer.decode(generation_output[0])
            runner.logger.info(f'Sample output:\n'
                               f'{inputs + generation_output}\n')
            if save_eval_output:
                eval_outputs.append(f'{inputs + generation_output}\n')

        if save_eval_output:
            self._save_eval_output(runner, eval_outputs)
