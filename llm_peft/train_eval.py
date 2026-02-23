from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer

from model_manager import download_model

def compute_metrics(predictions):
	# bring your own metrics computation code
	# 원하는 metrics 항목은 1중첩 flat dict 형태라면 어떤 값이든 상관 없으니 자유롭게 계산하세요.
	# 단, None, Null, NaN 등 은 불가능합니다! (플랫폼 화면에 표시 불가능. 0으로 치환하는 등 처리를 권장합니다.)
	metrics = {}
	return metrics

# 만약 compute_metrics 등 사용자 지정 metric 없이 기본값을 사용한다면,
# "model_preparation_time" 등을 기본제공되는 metric에서 제거하세요.

def trainer_build(model, data, configs):
	
	download_path = download_model(configs.model_configs.pretrained_model_name_or_path)
	# download_model 함수를 이용하여 minio로부터 다운로드 받으면, 이미 존재하는 경로는 다운로드 없이 경로만 반환합니다.
	tokenizer = AutoTokenizer.from_pretrained(download_path)
	# data_loader 에서 load 한 것처럼 tokenizer를 설정합니다.

	trainer = Trainer(
		model=model,
		args=TrainingArguments(**configs.trainer_configs),
		tokenizer=tokenizer,
		compute_metrics=compute_metrics,
		train_dataset=data["train"],
		eval_dataset=data["test"],
	)

	return trainer

# data = 여러분이 data_loader 함수에서 반환한 데이터 변수(보통 dict)

def train(model, data, configs):
	trainer = trainer_build(model, data, configs)
	trainer.train()
	# 학습시에는 metric을 반환하지 않습니다.

def evaluate(model, data, configs):
	trainer = trainer_build(model, data, configs)
	metrics : dict = trainer.evaluate()
	# 평가시에는 metric을 반환합니다.
	# 반환되는 metrics는 dict 형태여야만 합니다.
	return metrics
	# you can return only one dict variable!