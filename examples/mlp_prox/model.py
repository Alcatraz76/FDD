import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
	def __init__(self):
		super(MLP, self).__init__()
		self.fc1 = nn.Linear(28*28, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, 10)
		
	def forward(self,x):
		x = x.view(-1, 28*28)
		x = self.fc1(x)
		x = F.sigmoid(x)
		x = self.fc2(x)
		x = F.sigmoid(x)
		x = self.fc3(x)
		x = F.log_softmax(x, dim=1)
		return x
	
def model_loader(configs):
	model = MLP()
	return model