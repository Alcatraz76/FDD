from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

def data_loader(raw_datas, configs):
	# !! 원래 본 예제코드는 연합학습 환경에서 CNN 모델을 학습하는 것이었으므로, 데이터 또한 cifar10 데이터였으나,
	# 현재 플랫폼 출시 후 데이터는 KDA DB로 접근하는것으로 변경되었으며, 예제 데이터 활용은 불가능하고 ADMET 데이터 표준양식을 활용하도록 데이터 전처리를 하여야 합니다.
	# 따라서 본 예제파일 코드를 그대로 업로드하더라도 실행이 불가능합니다. 구조 참고용으로 활용해주세요.	
	
	# DB에 저장된 데이터는 별도로 분할세트가 없습니다. 사용자가 원하는 비율에 따라 분할하여 사용하세요.
	# 75대 25 분할
	train = raw_datas["solubility"].iloc[0: len(raw_datas["solubility"])*0.75]
	test = raw_datas["solubility"].iloc[len(raw_datas["solubility"])*0.75: ]

	# train_dataset = 사용자 전처리 진행
	# test_dataset = 사용자 전처리 진행
	
	# 꼭 데이터로더는 아니어도 좋습니다. 사용자가 받아 사용가능한 형태로 처리하세요
	train_loader = DataLoader(train_dataset, batch_size=4, num_workers=16)
	test_loader = DataLoader(test_dataset, batch_size=4, num_workers=16)

	# 반환은 언제나 하나의 딕셔너리 변수로 묶어 반환
	data = {"train": train_loader, "test": test_loader}
	return data