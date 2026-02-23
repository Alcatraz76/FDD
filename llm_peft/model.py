import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from omegaconf import DictConfig, OmegaConf

# 연합학습 클라이언트는 외부망 접속이 불가능합니다 (in/out bound 제한)

# from huggingface_hub import login
# login(token = 'token_token_token_token_token_token')

# 따라서 huggingface_hub 사용이 불가능하므로, 하단 용법에 따라 model_manager 모듈과 download_model 함수를 사용하세요

# model_manager.py is a custom module and it will be provided by the system automatically.
# don't worry, just import.
from model_manager import download_model


class CausalLMPEFTModel(nn.Module): 
	# nvflare model persistor에서는 클래스명 import를 통해 호출하므로, 인자전달 외에는 동적인 객체 호출이 불가능합니다
	# nvflare server : model.CausalLMPEFTModel() 형태로 호출, from pretrained 등 사용이 불가능
	# nvflare client : script.py 동작에 의존하므로, 일반적인 방법으로 모델 init, args 전달이 가능합니다.
	# nvflare model persistor와 nvflare 클라이언트의 모델 호출 형태가 동일할 수 있도록 (둘 다 from pretrained 사용이 가능하도록)
	# Wrapper 클래스를 생성하여 init시 자동으로 from pretrained를 진행하도록 합니다.
	
	def __init__(self, configs: DictConfig):
		# 사용자가 업로드한 configs.yaml 파일 내용을 모두 json >> dict 형태로 인자를 전달합니다.
		super(CausalLMPEFTModel, self).__init__()
		
		configs = OmegaConf.create(configs)
		# 호출 args로 주어진 configs를 애트리뷰트 타입으로 사용이 가능하도록 OmegaConf 패키지를 사용하여 변환합니다.
		
		download_path = download_model(configs.model_configs.pretrained_model_name_or_path)
		# huggingface 대용 기능입니다. FDD 플랫폼에서 제공하는 "사전학습 모델 파일" 업로드 기능이 Minio 기반입니다.
		# minio에 업로드한 모델 파일 또는 폴더를 다운로드 받고, 절대경로를 반환합니다. huggingface repo와 동일한 구조의 폴더를 업로드해주세요.

		peft_config = LoraConfig(**configs.model_configs.lora_config)

		full_model = AutoModelForCausalLM.from_pretrained(download_path)
		# 알고 계시겠지만, from_pretrained() 메서드는 huggingface_hub에도 접속이 가능하지만 로컬 디렉토리 경로를 주면 해당 디렉토리로부터 동일한 방식으로 load 합니다.
		# full_model을 self에 할당하면 파라미터 갯수(용량)가 두배! (peft_model 안에도 들어가니까)

		self.peft_model = get_peft_model(full_model, peft_config)
		# 표준 script의 가중치 load 방식은 state_dict() 이기 때문에, peft 사용시 적용이 어렵습니다.
		# 아래 용법에 따라 state_dict() 메서드 오버라이딩을 적용하세요.

	# trainable state_dict 반환 메서드 오버라이딩 -> 학습 가능한 파라미터만 반환, 모델 클래스의 state_dict 메서드를 이걸로 바꿉니다.
	# peft 또는 하나의 클래스 내 여러 모델 클래스를 사용하는 경우 등, 전체 레이어는 너무 heavy하고 특정 레이어만 전송하야 하는 경우에만 오버라이드를 사용하세요!
	def state_dict(self, *args, **kwargs):
		full_state_dict = super().state_dict(*args, **kwargs)
		trainable_keys = {name for name, param in self.named_parameters() if param.requires_grad}
		trainable_state_dict = {name: full_state_dict[name] for name in trainable_keys}
		return trainable_state_dict

	# 래퍼클래스의 forward
	def forward(self, *args, **kwargs):
		return self.peft_model(*args, **kwargs)

# 이 함수는 클라이언트에서만 사용합니다. 따라서 OmegaConf 객체를 인자로 받기때문에 별도 변환은 필요하지않습니다
def model_loader(configs: DictConfig):
	model = CausalLMPEFTModel(configs)
	return model