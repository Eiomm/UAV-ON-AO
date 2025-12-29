from datasets import load_dataset

# 下载并加载数据集
dataset = load_dataset('Kyaren/UAV-ON-dataset')

# 将数据集保存到本地
dataset.save_to_disk('./uav-on-dataset')
