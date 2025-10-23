import torch
import nvflare.client as flare
import time
import traceback

from training_manager import TrainingManager
from data import data_loader
from model import model_loader
from train_eval import train, evaluate

DEVICE = torch.device("cuda:0")

def main():
	# 1. 플레어 초기화
	flare.init()
	
	# 2. 트레이닝 매니저 초기화
	# 보안사항으로 인하여 training_manager.py는 제공되지 않습니다.
	train_manager = TrainingManager()

	try:
		err_code = 1000
		# 3. 사용자 설정파일 load
		# 사용자 설정파일 config.yaml 이 없을 경우, 로드되지 않으며 train_manager.configs = None 입니다.
		train_manager.set_config()

		# 4. 데이터 다운로드
		# KDA DB로부터 가능한 데이터를 가져옵니다.
		err_code = 2000
		raw_datas = train_manager.data_download() # -> Dict[str(key): DataFrame, ...]

		# 5. 사용자의 데이터 전처리 코드 호출
		err_code = 2001
		data = data_loader(raw_datas, train_manager.configs) # -> Dict[str(key): DataLoader, ...]

		err_code = 3000
		model = model_loader(train_manager.configs) # -> torch.nn.Module
	
	except Exception as e:
		train_manager.err(err_code, traceback.format_exc())

	# 8. nvflare 동작
	while flare.is_running():
		# 8.1. 모델 수신
		round_start = time.time()
		try:
			err_code = 4000 # flare 서버수신
			input_model = flare.receive()

			# 8.2. 현재라운드 설정
			train_manager.current_round = input_model.current_round+1
			train_manager.metric_dict[train_manager.current_round] = {}
			train_manager.time_check("round_start", round_start)

			# 8.3 수신 모델 파라미터 클라이언트 model 인스턴스에 로드
			err_code = 4001 # 서버 수신 모델 업데이트 오류
			model.load_state_dict(input_model.params)
			# model to DEVICE
			model.to(DEVICE)
		
			train_manager.time_check("receive_time", time.time())
			
			# 8.4. 글로벌 모델 수신 후 eval
			err_code = 5000 # 글로벌 모델 eval 오류
			train_manager.status = "glob_eval"
			train_manager.status_upload()
			train_manager.time_check("glob_eval_start", time.time())
			
			# 글로벌 모델 eval
			glob_metric = evaluate(model, data, train_manager.configs)
			
			train_manager.metric_save(glob_metric)
			train_manager.time_check("glob_eval_end", time.time())

			# 8.5. 학습
			err_code = 6000 # 학습 오류
			train_manager.status = "train"
			train_manager.status_upload()

			train_manager.time_check("train_start", time.time())

			train(model, data, train_manager.configs)
			
			train_manager.time_check("train_end", time.time())

			# 8.6. 로컬 학습 후 eval
			err_code = 7000 # 로컬 모델 eval 오류
			train_manager.status = "local_eval"
			train_manager.status_upload()
			train_manager.time_check("local_eval_start", time.time())

			# 로컬 모델 eval
			local_metric = evaluate(model, data, train_manager.configs)

			train_manager.metric_save(local_metric)
			train_manager.time_check("local_eval_end", time.time())

			# 8.7. 모델 전송
			err_code = 8000 # 모델 전송 오류
			train_manager.status = "send"
			train_manager.status_upload()

			output_model = flare.FLModel(params=model.cpu().state_dict())
			flare.send(output_model)
			train_manager.time_check("send_time", time.time())
			
			train_manager.time_check("round_end", time.time())
		
		except Exception as e:
			train_manager.err(err_code, traceback.format_exc())

	train_manager.status = "finished"
	train_manager.status_upload()

if __name__ == "__main__":
	main()