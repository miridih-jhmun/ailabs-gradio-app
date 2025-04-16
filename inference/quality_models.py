import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
from PIL import Image
import time
from transformers import AutoImageProcessor
from datetime import datetime
from transformers import Blip2VisionConfig
from models.blip_v2_model_inference import BlipV2ModelInference

# 글로벌 변수로 모델과 프로세서 관리
_model = None
_processor = None
_device = None
_model_path = '/app/data/element_low_quality_designhub_dataset/trained_model_weights/sojang-designhub-Light-BLIP-V2-250403'

backbone_name = 'BLIP-V2'
element_image_path = '/app/data/element_low_quality_designhub_dataset/images'
file_path = '/app/data/s3_dataset/file_miricanvas_com/raster_image_thumb/2024/09/10/16/30/kn9a234altz19oth/sticker_thumb_400.webp'
logit_scale_init_value = 2.6592
class_num = 2

def load_model():
    """필요할 때만 모델을 로드하는 함수"""
    global _model, _processor, _device
    
    if _model is None:
        try:
            print(f"[{datetime.now()}] BLIP-V2 모델 로드 시작...")
            
            # GPU 사용 가능 여부 확인
            _device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
            
            # 모델 설정 및 로드
            config = Blip2VisionConfig()
            config.logit_scale_init_value = logit_scale_init_value
            
            _model = BlipV2ModelInference.from_pretrained(
                f'{_model_path}/model_checkpoint.pth',
                _device,
                config=config,
                num_labels=class_num
            )
            
            # 프로세서 로드
            _processor = AutoImageProcessor.from_pretrained(_model_path, use_fast=True)
            
            # 모델을 평가 모드로 설정
            _model.to(_device)
            _model.eval()
            
            print(f"[{datetime.now()}] BLIP-V2 모델 로드 완료 (device: {_device})")
            return True, "모델이 로드되었습니다."
            
        except Exception as e:
            print(f"[{datetime.now()}] 모델 로드 중 오류 발생: {e}")
            _model = None
            _processor = None
            return False, f"모델 로드 실패: {str(e)}"
    
    return True, "모델이 이미 로드되었습니다."

def is_model_loaded():
    """모델이 로드되었는지 확인하는 함수"""
    global _model
    return _model is not None
def check_model_status():
    """
    모델 로드 상태를 확인하는 함수
    Returns:
        loaded (bool): 모델 로드 여부
        message (str): 상태 메시지
    """
    if is_model_loaded():
        return True, "모델이 로드되었습니다."
    else:
        return False, "모델이 로드되지 않았습니다. '모델 로드' 버튼을 클릭하세요."
    
def _read_image(image_path_or_obj) -> tuple:
    try:
        # 이미지 경로 또는 객체 처리
        if isinstance(image_path_or_obj, str):
            print(f"[{datetime.now()}] 이미지 경로: {image_path_or_obj}")
            img = Image.open(image_path_or_obj)
        else:
            print(f"[{datetime.now()}] 이미지 객체 사용")
            img = image_path_or_obj

        img = img.convert('RGBA')
        alpha_img = img.split()[-1]
        rgb_img = img.convert('RGB')
        
        return rgb_img, alpha_img, None
    except Exception as e:
        error_msg = f"이미지 로드 중 오류: {e}"
        print(error_msg)
        return None, None, error_msg

# 디자인 허브 측면 저퀄리티 분류 모델 -> quality_score, label
def design_hub_low_quality_inference(image_path, text=None) -> tuple:
    """
    디자인 허브 저퀄리티 분류 모델을 사용하여 이미지 품질을 평가
    
    Args:
        image: 이미지 파일 경로 또는 이미지 객체
        text: 텍스트 정보 (현재 사용하지 않음)
        
    Returns:
        (quality_score, label, error_message): 품질 점수(0-100), 라벨("승인"/"거부"), 오류 메시지(있는 경우)
    """
    # 모델을 로드할 수 없거나 오류 발생 시 임시로 랜덤 결과 반환
    model_load_result, error_msg = load_model()
    if not model_load_result:
        print(f"{error_msg}")
        quality_score = np.random.randint(1, 101)
        label = "승인" if quality_score > 50 else "거부"
        return quality_score, label, f"⚠️ {error_msg} (랜덤한 값을 반환합니다)"
    
    try:
        # 이미지 전처리
        start_time = time.time()
        rgb_image, alpha_image, error_msg = _read_image(image_path)
        
        if error_msg:
            raise ValueError(error_msg)
        
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
        return quality_score, label, None
        
    except Exception as e:
        error_msg = f"추론 중 오류 발생: {e}"
        print(error_msg)
        # 오류 발생 시 기본값 반환
        quality_score = np.random.randint(1, 101)
        label = "승인" if quality_score > 50 else "거부"
        return quality_score, label, f"⚠️ {error_msg} (랜덤한 값을 반환합니다)"

# AI 이미지 측면 저퀄리티 분류 모델
def ai_image_low_quality_inference(image, text=None):
    # 출력: 품질 점수, 라벨(AI/Non-AI), 에러 메시지
    # 예시: 랜덤으로 가끔 에러 발생시키기
    if np.random.random() < 0.3:  # 30% 확률로 에러 시뮬레이션
        quality_score = np.random.randint(1, 101)
        label = "AI" if quality_score > 50 else "Non-AI"
        return quality_score, label, "⚠️ AI 모델 로드 실패 (시뮬레이션된 에러)"
    
    quality_score = np.random.randint(1, 101)
    label = "AI" if quality_score > 50 else "Non-AI"
    return quality_score, label, None