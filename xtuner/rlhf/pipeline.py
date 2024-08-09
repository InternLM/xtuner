import argparse
import json
import os
import shutil
import time

from loguru import logger

from xtuner.rlhf.config.config import Config
from xtuner.rlhf.coordinator import Coordinator
from xtuner.rlhf.dataset import MessageIter
from xtuner.rlhf.envs import TxtEnv
from xtuner.rlhf.repeaters import KLGAERepeater
from xtuner.rlhf.timer import Timer
from xtuner.rlhf.trainer import PPOTrainer

import ray
from copy import deepcopy
from xtuner.rlhf.envs.utils import SYSTEM_PROMPT
from policy_output import (PolicyOutput, concat_policy_outputs,
                             logprobs_from_logits)

def parse_args():
    parser = argparse.ArgumentParser(description='Train LLM')
    parser.add_argument(
        '-c',
        '--config',
        help='config file name or path.',
        type=str,
        default='examples/rlhf/four_model_vllm_8gpu.py')
    parser.add_argument(
        '-w',
        '--work_dir',
        help='the dir to save logs and models',
        type=str,
        default=None)
    parser.add_argument(
        '-a', '--address', help='ray head address', type=str, default='auto')
    args = parser.parse_args()
    return args


def validate_config(config: Config):
    assert config['model_configs'] is not None
    assert config['model_configs']['policy'] is not None
    assert config['model_configs']['policy']['model_path'] is not None
    assert config['dataset_config'] is not None
    assert config['rollout_config'] is not None
    assert config['rollout_config']['generate_kwargs'] is not None
    assert config['rollout_config']['max_new_tokens'] is not None


def flatten_list(nested_list):
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened


class DataGenerator:
    def __init__(
        self,
        prompt_mes_iter,
        pretrain_mes_iter = None,
        resume_step=-1,
    ):
        self.prompt_mes_iter = iter(prompt_mes_iter)
        self.pretrain_mes_iter = iter(
            pretrain_mes_iter) if pretrain_mes_iter.message_datasets else None
        self.resume_step = resume_step

    def get(self):
        while self.resume_step > 0:
            logger.info(f'[Resume] {self.resume_step} consuming data...')
            next(self.prompt_mes_iter)
            if self.pretrain_mes_iter is not None:
                next(self.pretrain_mes_iter)
            self.resume_step -= 1

        # prompt data
        prompt_datas = deepcopy(next(self.prompt_mes_iter))
        prompt_input_messages = []
        for data in prompt_datas:
            assert data.mes_type == 'prompt'
            if data.sys_prompt != 'default':
                message = deepcopy([
                    dict(
                        role='system', content=SYSTEM_PROMPT[data.sys_prompt])
                ] + data.message)
            else:
                message = deepcopy(data.message)
            prompt_input_messages.append(message)

        # pretrain data
        if self.pretrain_mes_iter is not None:
            pretrain_input_messages = []
            pretrain_datas = deepcopy(next(self.pretrain_mes_iter))
            for data in pretrain_datas:
                assert data.mes_type == 'pretrain'
                pretrain_input_messages.append(message)
        
        if self.pretrain_mes_iter is not None:
            return prompt_datas, prompt_input_messages, pretrain_input_messages
        else:
            return prompt_datas, prompt_input_messages, None


if __name__ == '__main__':
    args = parse_args()
    assert args.config is not None, 'config should not be None'
    work_dir = args.work_dir
    if work_dir is None:
        work_dir = os.getcwd() + '/rlhf_trainlog_' + time.strftime(
            '%Y-%m-%d-%H:%M:%S')
    work_dir = os.path.abspath(work_dir)
    logger.info(f'using work_dir: {work_dir}')
    os.makedirs(work_dir, exist_ok=True)
    # save original config
    shutil.copy2(args.config, f'{work_dir}/{os.path.basename(args.config)}')
    logger.add(
        f'{work_dir}/train_rlhf.log',
        filter=lambda record: record['extra'].get('name') == 'train')
    logger_train = logger.bind(name='train')

    config = Config.from_file(args.config)
    logger.info('#################### CONFIG BGN ####################')
    for k, v in config.items():
        logger.info(f'{k}: {v}')
    logger.info('#################### CONFIG END ####################')

    # init model
    cluster_address = args.address
    if cluster_address != 'auto':
        cluster_address = f'ray://{cluster_address}:10001'
    logger.info(f'cluster_address={cluster_address}')
    coordinator = Coordinator(cluster_address, config)
    model_dict = coordinator.create_models()
    ref_model = model_dict['reference']
    policy_model = model_dict['policy']
    reward_model = model_dict['reward']
    critic_model = model_dict['critic']

    # init prompt & pretrain dataset
    prompt_dataset_config = config['prompt_dataset_config']
    prompt_mes_iter = MessageIter(
        tokenizer=ref_model.tokenizer, **prompt_dataset_config)
    pretrain_dataset_config = config.get('pretrain_dataset_config', {})
    pretrain_mes_iter = MessageIter(
        tokenizer=ref_model.tokenizer, **pretrain_dataset_config)

    # init txt env
    rollout_config = config.get('rollout_config', {})
    txt_env = ray.remote(TxtEnv).remote(
        policy_model=policy_model,
        reward_model=reward_model,
        **rollout_config,
    )
    # init repeater
    repeater_config = config.get('repeater_config', {})
    ppo_repeater = ray.remote(KLGAERepeater).remote(
        ref_model=ref_model,
        reward_model=reward_model,
        **repeater_config,
    )
    klgae_repeater = ray.remote(KLGAERepeater).remote(
        ref_model=ref_model,
        reward_model=reward_model,
        **repeater_config,
    )
    # init trainer
    train_config = config.get('train_config', {})
    ppo = ray.remote(PPOTrainer).remote(
        policy_model=policy_model,
        critic_model=critic_model,
        **train_config)
    
    critic_warmup_step = train_config['critic_warmup_step']
    save_interval = train_config['save_interval']
    max_train_step = train_config.get('max_train_step', float('inf'))
    resume_step = train_config.get('resume_step', -1)
    critic_warmup_step = min(critic_warmup_step,
                             critic_warmup_step - resume_step)
    pipe_micro_bs = train_config['pipe_micro_bs']

    # init log file
    json_f = open(f'{work_dir}/train_rlhf.log.jsonl', 'w')

    data_generator = DataGenerator(
        prompt_mes_iter=prompt_mes_iter,
        pretrain_mes_iter=pretrain_mes_iter,  # None
        resume_step=resume_step,
    )

    step = max(0, resume_step)
    while step <= max_train_step:
        s_t = time.time()
        with Timer(f'step {step}: end_to_end'):
            # Get Data
            prompt_datas, prompt_input_messages, pretrain_input_messages = data_generator.get()

            # Create placeholder lists to manage intermediate results
            num_batch = len(prompt_input_messages) // pipe_micro_bs
            logger.info(f'prompt_bs={len(prompt_input_messages)}, '
                        f'pipe_micro_bs={pipe_micro_bs}, '
                        f'num_batch={num_batch}')

            traj_refs_stage1 = [None] * num_batch
            traj_refs_stage2 = [None] * num_batch
            reward_reference_stage2 = [None] * num_batch

            critic_loss_refs = [None] * num_batch
            policy_loss_refs = [None] * num_batch
            pretrain_loss_refs = [None] * num_batch

            # Stage 1: Generate trajectories
            txt_env.rollout_background.remote(prompt_input_messages, pretrain_input_messages)
            for idx in range(num_batch):
                traj_ref = txt_env.rollout_get.remote(pipe_micro_bs)
                traj_refs_stage1[idx] = traj_ref

            # Stage 2: Reward & Reference Model infer
            for idx in range(num_batch):
                traj_ref = traj_refs_stage1[idx]
                reward_ref, reference_logprobs_ref = (
                    ppo_repeater.get_reward_and_reference.options(num_returns=2).remote(
                        prompt_datas, 
                        traj_ref)
                )
                reward_reference_stage2[idx] = (reward_ref, reference_logprobs_ref)

            # Stage 3: Critic & Policy infer and learn
            for idx in range(num_batch):
                # Infer
                traj_ref = traj_refs_stage1[idx]
                values_ref, policy_logprobs_ref = ppo.infer.options(num_returns=2).remote(
                    traj_ref)

                # Process KL, GAE
                reward_ref, reference_logprobs_ref = reward_reference_stage2[idx]
                traj_ref_2 = klgae_repeater.process_kl_gae.remote(
                    reward_ref, 
                    reference_logprobs_ref, 
                    values_ref, 
                    policy_logprobs_ref, 
                    traj_ref)
                traj_refs_stage2[idx] = traj_ref_2

                # Train
                update_param = (idx == num_batch - 1)
                policy_loss_ref, pretrain_loss_ref, critic_loss_ref = (
                    ppo.train.options(num_returns=3).remote(
                        traj_ref_2, 
                        update_param, 
                        critic_warmup_step
                    )
                )
                critic_loss_refs[idx] = critic_loss_ref
                policy_loss_refs[idx] = policy_loss_ref
                pretrain_loss_refs[idx] = pretrain_loss_ref

            # Collect results
            policy_losses = flatten_list(ray.get(policy_loss_refs))
            pretrain_losses = flatten_list(ray.get(pretrain_loss_refs))
            critic_losses = flatten_list(ray.get(critic_loss_refs))
            ray.get(ppo.sync_model.remote())
            trajectories = ray.get(traj_refs_stage2)
            # Post process output
            padding_token_map = {'output_ids': policy_model.tokenizer.pad_token_id}
            trajectories = concat_policy_outputs(trajectories, 
                                                 padding_token_map)

        critic_warmup_step -= 1
        total_time = time.time() - s_t

        # write log
        if config['rollout_config'].get('write_to_file', True):
            if not os.path.exists(f'{work_dir}/rollouts'):
                os.makedirs(f'{work_dir}/rollouts')
            with open(f'{work_dir}/rollouts/step{step}_rollout.log',
                      'w') as file:
                for output_s, r, req_id in zip(trajectories.output_str, 
                                               trajectories.rewards, 
                                               trajectories.req_ids):
                    file.write(output_s + '\n' + 
                                'Reward: ' + str(r.item()) + '\n' + 
                                'Req_id: ' + str(req_id) + '\n' + 
                                '=' * 30 + '\n')

        policy_loss_mean = sum(policy_losses) / len(policy_losses) if policy_losses else None
        pretrain_loss_mean = sum(pretrain_losses) / len(pretrain_losses) if pretrain_losses else None
        critic_loss_mean = sum(critic_losses) / len(critic_losses) 

        summaries = dict(
            reward_mean=trajectories.rewards.mean().item(),
            reward_std=trajectories.rewards.std().item(),
            new_tokens_mean=trajectories.action_mask.sum(
                -1).float().mean().item(),
            new_tokens_std=trajectories.action_mask.sum(
                -1).float().std().item(),
            resp_tokens_mean=trajectories.answer_mask.sum(
                -1).float().mean().item(),
            kl=trajectories.kl.mean().item(),
            entropy=trajectories.entropy.mean().item(),
            step=step,
            policy_loss=policy_loss_mean,
            pretrain_loss=pretrain_loss_mean,
            critic_loss=critic_loss_mean,
            total_time=total_time,
        )
        json_f.write(json.dumps(summaries) + '\n')
        json_f.flush()
        logger_train.info(f'[end to end] duration: {time.time() - s_t} s')

        step += 1
        if (step % save_interval == 0) or (step == max_train_step):
            policy_model.save(f'{work_dir}/ckpt/policy_model/{step}')
            critic_model.save(f'{work_dir}/ckpt/critic_model/{step}')

    json_f.close()