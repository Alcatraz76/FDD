해당 파일은 사용자 코드를 받아 job 생성시 플랫폼 서버에서 자동으로 추가하는 파일입니다.
항상 플랫폼 서버 기준 최신화된 스크립트 코드를 제공하며, 덮어씌우기 때문에 사용자가 script 등을 수정하여도 반영이 불가능합니다.

총 4개 파일이 존재하며 그 중 2개의 파일은 보안상의 사유로 업로드 하지 않습니다.

script.py : 클라이언트 내 연합학습 진행 parent process. 최초 script부터 실행되며, 사용자코드를 import 하고 정해진 동작을 수행합니다.
controller_manager.py : 연합학습 시 사용되는 집계함수(Nvflare controller)를 위해, fedprox, scaffold 등을 지원하기 위하여 제공.
