import os
import re
import argparse
import json
import matplotlib.pyplot as plt

def extract_xtuner_rewards(folder_path):
    import os
    import glob
    from pathlib import Path
    matching_files = list(Path(folder_path).glob('rollout_idx_*.jsonl'))
    file_count = len(matching_files)
    # print("matching_files: ", matching_files)
    steps = []
    rewards = []
    
    file_count = len(matching_files)
    for i in range(1, file_count+1):
        file_path = os.path.join(folder_path, f'rollout_idx_{i}_trajectory.jsonl')
        print(file_path)
        reward = calculate_average_reward(file_path)
        steps.append(i)
        rewards.append(reward)
    return steps, rewards

def calculate_average_reward(file_path):
    all_rewards = []
    line_count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                if not line.strip():
                    continue              
                try:
                    data = json.loads(line)
                    if 'reward' in data and isinstance(data['reward'], list):
                        all_rewards.extend(data['reward'])
                except json.JSONDecodeError:
                    print(f"警告: 第 {line_count} 行不是有效的JSON格式，已跳过。")

    except FileNotFoundError:
        return

    total_reward_sum = sum(all_rewards)
    total_reward_count = len(all_rewards)
    average_reward = total_reward_sum / total_reward_count
    return average_reward

def extract_accuracy(folder_path):
    infer_log_path = folder_path + "/rank0.log"
    extract_accuracy_list = []
    accuracy_step = []
    with open(infer_log_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(r"'accuracy':\s*([\d.]+)", line)
            if match:
                accuracy_value = float(match.group(1))
                step_match = re.search(r"idx (.*?) scores", line)
                idx_value = step_match.group(1).strip() if step_match else None
                extract_accuracy_list.append(accuracy_value)
                accuracy_step.append(idx_value)
    return accuracy_step[1:], extract_accuracy_list[1:]

def extract_entropy(folder_path):
    train_log_path = folder_path + "/train_rank0.log"
    entropy_list = []
    with open(train_log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "entropy" in line:
                entropy = float(line.split(":")[4].strip())
                entropy_list.append(entropy)
    return entropy_list

def extract_grad_norm(folder_path):
    train_log_path = folder_path + "/train_rank0.log"
    grad_norm_list = []
    with open(train_log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "grad_norm" in line:
                print(line)
                grad_norm = float(line.split("=")[4].strip())
                grad_norm_list.append(grad_norm)
    return grad_norm_list


def plt_image(step, reward, entropy, grad_norm, loss, eval_step, eval_accuracy, output_path):
    plt.figure(figsize=(14, 8))

    # Plot each metric
    if reward:
        plt.plot(step, reward, marker='o', linestyle='-', markersize=4, label='Reward')
    if grad_norm:
        plt.plot(step, grad_norm, marker='x', linestyle=':', markersize=4, label='Gradient Norm')
    if loss:
        plt.plot(step, loss, marker='d', linestyle='-', markersize=4, label='Loss')
    if entropy:
        plt.plot(step, entropy, marker='s', linestyle='--', markersize=4, label='Entropy')
    # if eval_accuracy:
    #     plt.scatter(eval_step, eval_accuracy, marker='o', color='r', label='EvalAccuracy', zorder=5)
    # Add chart title and axis labels
    plt.title('Training Metrics Over Steps', fontsize=16)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)

    # Add a grid for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Add a legend
    plt.legend()

    # Adjust layout automatically
    plt.tight_layout()

    # Save the chart to a file
    plt.savefig(output_path)
    print(f"Chart saved to: {output_path}")

    # Display the chart
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot rewards from XTuner log files.")
    parser.add_argument("--log-dir-path", type=str, help="Path to the directory containing xtuner log files.")
    args = parser.parse_args()

    xtuner_steps, xtuner_reward = extract_xtuner_rewards(args.log_dir_path)
    eval_step, eval_accuracy = extract_accuracy(args.log_dir_path)
    entropy = extract_entropy(args.log_dir_path)
    grad_norm = extract_grad_norm(args.log_dir_path)
    save_path = os.path.join(args.log_dir_path, "xtuner_rl_metrics.png")
    len = min(len(xtuner_steps), len(xtuner_reward), len(entropy), len(grad_norm))
    # plt_image(xtuner_steps[:len], xtuner_reward[:len], entropy[:len], grad_norm[:len], None, eval_step, eval_accuracy, save_path)
    plt_image(xtuner_steps[:len], xtuner_reward[:len], None, None, None, None, None, save_path)
