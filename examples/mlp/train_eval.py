import torch
import torch.nn as nn

DEVICE = torch.device("cuda:0")
EPOCHS = 1
LEARNING_RATE = 0.001

def train(model, data, configs):
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

def evaluate(model, data, configs):
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
	metrics = {"loss": test_loss, "accuracy": test_accuracy}
	return metrics
