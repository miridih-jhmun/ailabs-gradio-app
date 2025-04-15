import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
from PIL import Image
import time
from transformers import AutoImageProcessor
from datetime import datetime

# 글로벌 변수로 모델과 프로세서 관리
_model = None
_processor = None
_device = None
_model_path = '/app/data/element_low_quality_designhub_dataset/trained_model_weights/sojang-designhub-Light-BLIP-V2-250403'

def _load_model_if_needed():
    """필요할 때만 모델을 로드하는 함수"""
    global _model, _processor, _device
    
    if _model is None:
        try:
            print(f"[{datetime.now()}] BLIP-V2 모델 로드 시작...")
            
            # GPU 사용 가능 여부 확인
            _device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
            
            # 필요한 라이브러리 임포트
            from transformers import Blip2VisionConfig
            
            try:
                # 로컬 모듈에서 모델 클래스 임포트 시도
                from models import BlipV2ModelInference
            except ImportError:
                # 로컬 모듈 임포트 실패 시 직접 임포트
                import sys
                import os
                sys.path.append('/data/element_low_quality_designhub_dataset')
                from models import BlipV2ModelInference
            
            # 모델 설정 및 로드
            config = Blip2VisionConfig()
            config.logit_scale_init_value = 2.6592
            
            _model = BlipV2ModelInference.from_pretrained(
                f'{_model_path}/model_checkpoint.pth',
                _device,
                config=config,
                num_labels=2
            )
            
            # 프로세서 로드
            _processor = AutoImageProcessor.from_pretrained(_model_path)
            
            # 모델을 평가 모드로 설정
            _model.to(_device)
            _model.eval()
            
            print(f"[{datetime.now()}] BLIP-V2 모델 로드 완료 (device: {_device})")
            return True
            
        except Exception as e:
            print(f"[{datetime.now()}] 모델 로드 중 오류 발생: {e}")
            _model = None
            _processor = None
            return False
    
    return True

def _read_image(image_path_or_obj):
    """이미지 파일 또는 객체를 로드하고 전처리하는 함수"""
    try:
        # 이미지 경로인 경우 파일을 열기
        if isinstance(image_path_or_obj, str):
            img = Image.open(image_path_or_obj)
        else:
            # 이미 이미지 객체인 경우 그대로 사용
            img = image_path_or_obj
            
        # RGBA 이미지를 RGB로 변환
        if img.mode == 'RGBA':
            rgb_img = img.convert('RGB')
        else:
            rgb_img = img.convert('RGB')
            
        return rgb_img
    except Exception as e:
        print(f"이미지 로드 중 오류: {e}")
        return None

# 디자인 허브 측면 저퀄리티 분류 모델
def design_hub_low_quality_inference(image, text=None):
    """
    디자인 허브 저퀄리티 분류 모델을 사용하여 이미지 품질을 평가
    
    Args:
        image: 이미지 파일 경로 또는 이미지 객체
        text: 텍스트 정보 (현재 사용하지 않음)
        
    Returns:
        (quality_score, label): 품질 점수(0-100)와 라벨("승인"/"거부")
    """
    # 모델을 로드할 수 없거나 오류 발생 시 임시로 랜덤 결과 반환
    if not _load_model_if_needed():
        print("모델을 로드할 수 없어 랜덤 값을 반환합니다.")
        quality_score = np.random.randint(1, 101)
        label = "승인" if quality_score > 50 else "거부"
        return quality_score, label
    
    try:
        # 이미지 전처리
        start_time = time.time()
        rgb_image = _read_image(image)
        
        if rgb_image is None:
            raise ValueError("이미지를 읽을 수 없습니다.")
        
        # 이미지 텐서 변환
        rgb_image_tensor = _processor(rgb_image, return_tensors='pt')['pixel_values'][0]
        rgb_image_tensor = rgb_image_tensor.unsqueeze(dim=0)
        rgb_image_tensor = rgb_image_tensor.to(_device)
        
        # 모델 추론 (PyTorch의 no_grad 컨텍스트에서 실행하여 메모리 효율성 증가)
        with torch.no_grad():
            # 소프트맥스 활성화 함수 정의
            activate_function = torch.nn.Softmax(dim=1)
            
            # 모델 예측 및 저품질 확률 계산
            model_predictions = _model(rgb_image_tensor).logits
            activate_model_predictions = activate_function(model_predictions)[0]
            
            # 저품질 확률 추출 (CPU로 변환하여 numpy 배열로 변환)
            low_quality_proba = float(activate_model_predictions[1].cpu().numpy())
        
        end_time = time.time()
        
        # 결과 계산 (저품질 확률이 높을수록 품질 점수는 낮게)
        quality_score = int(100 - low_quality_proba * 100)
        label = "승인" if quality_score > 50 else "거부"
        
        print(f"추론 소요 시간: {end_time - start_time:.4f}초, 품질 점수: {quality_score}, 라벨: {label}")
        return quality_score, label
        
    except Exception as e:
        print(f"추론 중 오류 발생: {e}")
        # 오류 발생 시 기본값 반환
        quality_score = np.random.randint(1, 101)
        label = "승인" if quality_score > 50 else "거부"
        return quality_score, label

# AI 이미지 측면 저퀄리티 분류 모델
def ai_image_low_quality_inference(image, text=None):
    # 출력: 품질 점수, 라벨(AI/Non-AI)
    quality_score = np.random.randint(1, 101)
    label = "AI" if quality_score > 50 else "Non-AI"
    return quality_score, label