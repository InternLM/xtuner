import os

import torch


# use global variable to avoid repeated os.getenv calls
DISTRIBUTED_COMMUNICATION_SM = int(os.getenv("DISTRIBUTED_COMMUNICATION_SM", "24"))
DEEPEP_SM = int(os.getenv("DEEPEP_SM", "20"))

NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
