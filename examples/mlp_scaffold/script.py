import torch
import nvflare.client as flare
import time
import traceback
from training_manager import TrainingManager, Status
from model_manager import *
from controller_manager import *
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

	# init 구간 전체는 status="Initial"(TrainingManager 생성 시점 값) 하에서 실행.
	# 이 구간에서 죽으면 err_name="Initial", 정확한 지점은 err_message(traceback)로 식별.
	try:
		# 3. 사용자 설정파일 load
		# 사용자 설정파일 config.yaml 이 없을 경우, 로드되지 않으며 train_manager.configs = None 입니다.
		train_manager.status = Status.SET_CONFIG
		train_manager.set_config()

		# 4. 데이터 다운로드
		# KDA DB로부터 가능한 데이터를 가져옵니다.
		train_manager.status = Status.DATA_DOWNLOAD
		raw_datas = train_manager.data_download() # -> Dict[str(key): DataFrame, ...]

		# 5. 사용자의 데이터 전처리 코드 호출
		print(f"---------- Client side init : User data processing ----------")
		train_manager.status = Status.USER_DATA_LOAD
		data = data_loader(raw_datas, train_manager.configs) # -> Dict[str(key): DataLoader, ...]

		print(f"---------- Client side init : User model loading ----------")
		train_manager.status = Status.USER_MODEL_LOAD
		model = model_loader(train_manager.configs) # -> torch.nn.Module

		print(f"---------- Client side init : User model stats ----------")
		print_stats(model_stats(model))

	except Exception as e:
		train_manager.err(traceback.format_exc())

	# 8. nvflare 동작
	while flare.is_running():
		# 8.1. 모델 수신
		round_start = time.time()
		try:
			# 서버 모델 수신 ~ 가중치 로드 구간. 이 시점에 죽으면 err_name="round_start".
			train_manager.status = Status.ROUND_START
			input_model = flare.receive()

			# 8.2. 현재라운드 설정
			train_manager.current_round = input_model.current_round+1
			train_manager.metric_dict[train_manager.current_round] = {}
			train_manager.time_check("round_start", round_start)
			print(f"---------- Round {train_manager.current_round} : start ----------")

			# 8.3 수신 모델 파라미터 클라이언트 model 인스턴스에 로드
			print(f"---------- Round {train_manager.current_round} : Recived global model stats ----------")
			recive_model_stats(input_model)

			print(f"---------- Round {train_manager.current_round} : Recived global model --> local model weight load ----------")
			train_manager.status = Status.LOAD_STATE_DICT
			result = model.load_state_dict(input_model.params, strict=False)
			print("missing:", len(result.missing_keys))
			print("unexpected:", len(result.unexpected_keys))
			
			# model to DEVICE
			train_manager.status = Status.MODEL_TO_DEVICE
			model = model.to(DEVICE)
			print(f"---------- Round {train_manager.current_round} : Recived model to {DEVICE} ----------")
		
			train_manager.time_check("receive_time", time.time())
			
			# 8.4. 글로벌 모델 수신 후 eval
			train_manager.status = Status.GLOB_EVAL
			train_manager.status_upload()
			train_manager.time_check("glob_eval_start", time.time())
			print(f"---------- Round {train_manager.current_round} : Global eval start ----------")
			
			# 글로벌 모델 eval
			glob_metric = evaluate(model, data, train_manager.configs)
			
			train_manager.metric_save(glob_metric)
			train_manager.time_check("glob_eval_end", time.time())
			print(f"---------- Round {train_manager.current_round} : Global eval end ----------")

			# 8.5. 학습
			train_manager.status = Status.TRAIN
			train_manager.status_upload()

			train_manager.time_check("train_start", time.time())
			print(f"---------- Round {train_manager.current_round} : Train start ----------")
			print(f"---------- Round {train_manager.current_round} : Controller : {train_manager.controller} ----------")

			# scaffold
			if train_manager.controller == "scaffold":
				fl_scaffold = FLScaffold(model)
				fl_scaffold.get_global_controls(input_model)
			
			train(model, data, train_manager.configs)

			train_manager.time_check("train_end", time.time())
			print(f"---------- Round {train_manager.current_round} : Train end ----------")
			# 8.6. 로컬 학습 후 eval
			train_manager.status = Status.LOCAL_EVAL
			train_manager.status_upload()
			train_manager.time_check("local_eval_start", time.time())
			print(f"---------- Round {train_manager.current_round} : Local eval start ----------")

			# 로컬 모델 eval
			local_metric = evaluate(model, data, train_manager.configs)

			train_manager.metric_save(local_metric)
			train_manager.time_check("local_eval_end", time.time())
			print(f"---------- Round {train_manager.current_round} : Local eval end ----------")

			# 8.7. 모델 전송
			train_manager.status = Status.SEND
			train_manager.status_upload()
			print(f"---------- Round {train_manager.current_round} : Try to send model ----------")

			# scaffold
			if train_manager.controller == "scaffold":
				output_model = flare.FLModel(params=model.cpu().state_dict(), meta=fl_scaffold.scaffold_meta())
			else:
				output_model = flare.FLModel(params=model.cpu().state_dict())

			flare.send(output_model)
			train_manager.time_check("send_time", time.time())
			print(f"---------- Round {train_manager.current_round} : Model sent ----------")

			train_manager.time_check("round_end", time.time())
		
		except Exception as e:
			print(f"---------- Round {train_manager.current_round} : Error ----------")
			train_manager.err(traceback.format_exc())

	train_manager.status = Status.FINISHED
	train_manager.status_upload()

if __name__ == "__main__":
	main()