from datasets import load_dataset
from mmengine.dataset import DefaultSampler

from mmchat.datasets import process_hf_dataset
from mmchat.datasets.collate_fns import default_collate_fn
from mmchat.datasets.map_fns import cmd_map_fn

data_root = './data/'

# Download data from https://github.com/Toyhom/Chinese-medical-dialogue-data
all_csv = [
    'Chinese-medical-dialogue-data/Data_数据/Andriatria_男科/男科5-13000.csv',
    'Chinese-medical-dialogue-data/Data_数据/IM_内科/内科5000-33000.csv',
    'Chinese-medical-dialogue-data/Data_数据/OAGD_妇产科/妇产科6-28000.csv',
    'Chinese-medical-dialogue-data/Data_数据/Oncology_肿瘤科/肿瘤科5-10000.csv',
    'Chinese-medical-dialogue-data/Data_数据/Pediatric_儿科/儿科5-14000.csv',
    'Chinese-medical-dialogue-data/Data_数据/Surgical_外科/外科5-14000.csv'
]

all_csv = [data_root + csv for csv in all_csv]

cmd = dict(
    type=load_dataset,
    path='csv',
    data_files=dict(train=all_csv),
    encoding='GB18030')

train_dataset = dict(
    type=process_hf_dataset,
    dataset=cmd,
    split='train',
    tokenizer=None,
    max_length=2048,
    map_fn=cmd_map_fn,
    remove_columns=[],
    concat_to_max_length=True)

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))
