import torch
import torch.nn as nn

DEVICE = torch.device("cuda:0")
EPOCHS = 1
LEARNING_RATE = 0.001

def train(model, data, configs):
	# script로부터 전달받은 data 딕셔너리 내에는 사용자가 반환하였던 데이터가 저장되어있습니다.
	train_data = data["train"]

	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
	criterion = nn.CrossEntropyLoss()

	for epoch in range(EPOCHS):
		running_loss = 0.0
		model.train()
		for batch_idx, (image, label) in enumerate(train_data):
			image = image.to(DEVICE)
			label = label.to(DEVICE)
			optimizer.zero_grad()
			predictions = model(image)
			cost = criterion(predictions, label)
			cost.backward()
			optimizer.step()
			running_loss += cost.cpu().detach().numpy() / image.size()[0]

	# 연합학습 시 train 단계에서는 metric을 수집하지 않습니다.

def evaluate(model, data, configs):
	# script로부터 전달받은 data 딕셔너리 내에는 사용자가 반환하였던 데이터가 저장되어있습니다.
	test_data = data["test"]

	criterion = nn.CrossEntropyLoss()
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for image, label in test_data:
			image = image.to(DEVICE)
			label = label.to(DEVICE)
			output = model(image)
			test_loss += criterion(output, label).item()
			prediction = output.max(1, keepdim=True)[1]
			correct += prediction.eq(label.view_as(prediction)).sum().item()
			
	test_loss = test_loss / len(test_data.dataset)
	test_accuracy = 100. * correct / len(test_data.dataset)

	# 사용자가 계산한 metirc을 data처럼 metrics 딕셔너리에 저장하여 반환합니다.
	# 사용자 정의 metric을 추가하여 사용 가능하지만, huggingface 또는 pytorch-lightning 등 trainer를 사용하는 경우
	# 대부분 metirc 반환형태가 flat한 dictionary 형태가 아니므로, 꼭 아래 형태로 변환하여 반환하세요
	metrics = {"loss": test_loss, "accuracy": test_accuracy}
	return metrics
