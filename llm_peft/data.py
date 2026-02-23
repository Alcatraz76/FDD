
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
from omegaconf import DictConfig

# model_manager.py is a custom module and it will be provided by the system automatically.
# don't worry, just import.
from model_manager import download_model

class CustomDataset(Dataset):
	def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int):
		# CustomDataset class for "pd.DataFrame -> Dataset"
		# bring your own code
		pass

def data_loader(raw_datas : dict, configs : DictConfig):
	# raw_datas 는 {task_type: pd.DataFrame, ...} 형태로 전달됩니다.
	# 원하는 task 키값을 골라 사용합니다.
	task_data : pd.DataFrame = raw_datas[configs.data_configs.task_type]

	download_path = download_model(configs.model_configs.pretrained_model_name_or_path)
	# download_model 함수를 이용하여 minio로부터 다운로드 받으면, 이미 존재하는 경로는 다운로드 없이 경로만 반환합니다.
	tokenizer = AutoTokenizer.from_pretrained(download_path)
	# tokenizer도 모델과 같이 huggingface가 아닌 minio를 통해 받습니다.

	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	max_length = configs.data_configs.max_length
	train_size = int(len(task_data) * 0.8)
	train_data = task_data.iloc[:train_size]
	test_data = task_data.iloc[train_size:]

	train_dataset = CustomDataset(train_data, tokenizer, max_length)
	test_dataset = CustomDataset(test_data, tokenizer, max_length)

	data = {"train": train_dataset, "test": test_dataset}
	# 반환은 언제나 하나의 변수로 묶어 반환
	return data
	# you can return only one variable!