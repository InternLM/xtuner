# 离线处理 Llava 训练数据集

当训练数据量非常大时，每次训练的时候都先在线处理数据可能会极为耗时。我们可以先对原始数据进行离线处理并保存至本地，随后的多次训练可以读入本地离线处理好的数据后直接开始训练。

## Step 1, 导出模板 config 文件

可使用以下命令查看 XTuner 中提供的 Llava 训练相关的 config：

```
xtuner list-cfg -p llava
```

找到需要使用的 config 文件并导出至当前目录下：

```
xtuner copy-cfg ${CONFIG_NAME} .
```

## Step 2, 离线处理数据集

使用以下命令可离线处理训练数据集中的文本数据：

```
python xtuner/tools/process_untokenized_llava_data.py \
    ${CONFIG_PATH} \
    --save-folder /folder/to/save/processed/dataset
```

其中，${CONFIG_PATH} 为第一步中导出的 config 文件路径，`/folder/to/save/processed/dataset` 则需要指定为离线处理数据的保存路径。

## Step 3, 修改 config 文件

对 Step 1 中导出的 config 文件做如下修改：

```diff
#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
llava_dataset = dict(
-   data_path=data_path,
-   tokenizer=tokenizer,
+   offline_processed_text_folder=/folder/to/save/processed/dataset
    ...)
```

其中，`/folder/to/save/processed/dataset` 为 Step 2 保存的离线处理数据路径。

## Step 4，开始训练

使用 Step 3 修改得到的 config 训练即可。
