import torch


class config:
    config = {
        "batch_size": 1024,  # 배치 사이즈 설정
        "device": (
            torch.device("cuda") if torch.cuda.is_available(
            ) else torch.device("cpu")
        ),  # GPU 사용 여부 설정
        "input_size": 16,  # 입력 차원 설정
        "hidden_size": 8,  # Hidden 차원 설정
        "output_size": 16,  # 출력 차원 설정
        "dropout": 0.2,  # Dropout 비율 설정
        "window_size": 60,  # sequence Lenght
        "num_layers": 2,  # LSTM layer 갯수 설정
        "lr": 0.05,  # learning rate 설정
        "epochs": 10,  # 총 반복 횟수 설정
        "early_stop": True,  # valid loss가 작아지지 않으면 early stop 조건 설정
    }
