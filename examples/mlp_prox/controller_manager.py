import torch
import copy

from nvflare.app_common.app_constant import AlgorithmConstants
from nvflare.app_opt.pt.fedproxloss import PTFedProxLoss
from nvflare.app_opt.pt.scaffold import PTScaffoldHelper

from model_manager import get_lr_values, DEVICE, compute_model_diff

class FLScaffold:
	_instance = None

	def __new__(cls, model):
		if cls._instance is None:
			cls._instance = super().__new__(cls)
			cls._instance.scaffold_helper = PTScaffoldHelper()
			cls._instance.scaffold_helper.init(model)
		return cls._instance

	def reset(cls):
		cls._instance = None

	def set_global(self, model):
		self.global_model = copy.deepcopy(model)
		for param in self.global_model.parameters():
			param.requires_grad = False

	def get_global_controls(self, input_model):
		if AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL not in input_model.meta:
			raise ValueError(
				f"Expected model meta to contain AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL "
				f"but meta was {input_model.meta}.",
			)
		
		global_ctrl_weights = input_model.meta.get(AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL)
		
		if not global_ctrl_weights:
			raise ValueError("global_ctrl_weights were empty!")
		# convert to tensor and load into c_global model
		
		for k in global_ctrl_weights.keys():
			global_ctrl_weights[k] = torch.as_tensor(global_ctrl_weights[k]).to(DEVICE)
		
		self.scaffold_helper.load_global_controls(weights=global_ctrl_weights)

	def get_controls(self):
		self.c_global_para, self.c_local_para = self.scaffold_helper.get_params()

	def scaffold_apply(self, model, optimizer):
		self.curr_lr = get_lr_values(optimizer)[0]  # Update learning rate after optimizer step
		self.scaffold_helper.model_update(model=model, curr_lr=self.curr_lr, c_global_para=self.c_global_para, c_local_para=self.c_local_para)

	def scaffold_update(self, model, steps):
		self.steps = steps
		self.scaffold_helper.terms_update(model=model, curr_lr=self.curr_lr, c_global_para=self.c_global_para, c_local_para=self.c_local_para, model_global=self.global_model)
		self.scaffold_diff(model)

	def scaffold_diff(self, model):
		self.model_diff, self.diff_norm = compute_model_diff(model, self.global_model)

	def scaffold_meta(self):
		meta={"NUM_STEPS_CURRENT_ROUND": self.steps,
				AlgorithmConstants.SCAFFOLD_CTRL_DIFF: self.scaffold_helper.get_delta_controls()}
		return meta

class FLFedProx:
	_instance = None

	def __new__(cls, fedproxloss_mu):
		if cls._instance is None:
			cls._instance = super().__new__(cls)
			cls._instance.fedproxloss_mu = fedproxloss_mu
			cls._instance.criterion_prox = cls._instance.fedprox_init()
		return cls._instance

	def reset(cls):
		cls._instance = None

	def fedprox_init(self):
		if self.fedproxloss_mu > 0:
			print(f"using FedProx loss with mu {self.fedproxloss_mu}")
			return PTFedProxLoss(mu=self.fedproxloss_mu)
		
	def set_global(self, model):
		self.global_model = copy.deepcopy(model)
		for param in self.global_model.parameters():
			param.requires_grad = False

	def fedprox_apply(self, model, loss):
		if self.fedproxloss_mu > 0 :
			fed_prox_loss = self.criterion_prox(model, self.global_model)
			loss += fed_prox_loss
		return loss