from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

... # 전처리 과정에서 호출할 함수 정의 자유

# 원본 데이터의 데이터프레임을 전달받고, 사용자 모델에 맞게 전처리하여 데이터 로더를 반환할 data_loader 함수 정의
def data_loader(raw_datas, configs):
	train_loader = DataLoader(raw_datas['train'], batch_size=32, shuffle=True, num_workers=4)
	test_loader = DataLoader(raw_datas['test'], batch_size=32, shuffle=True, num_workers=4)
	
	# 반환은 언제나 하나의 딕셔너리 변수로 묶어 반환
	data = {"train": train_loader, "test": test_loader}
	return data

	