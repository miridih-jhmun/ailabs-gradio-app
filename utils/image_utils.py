import os
from PIL import Image
from datetime import datetime
import numpy as np

def resize_image(image):
    """이미지 리사이징 함수 - NumPy 배열을 사용"""
    if image is None:
        return None
        
    # 원본 이미지 크기 가져오기
    width, height = image.size
    
    # 1/3 크기로 조정 (비율 유지)
    new_width = width // 3
    new_height = height // 3
    
    # 이미지를 NumPy 배열로 변환
    img_array = np.array(image)
    
    # 이미지가 RGB인지 확인
    print(f"[{datetime.now()}] 리사이징 전 이미지 배열 형태: {img_array.shape}, 타입: {img_array.dtype}")
    
    # PIL의 resize 메서드 사용 - 가장 기본적인 방법
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # NumPy 배열로 변환하여 값 확인
    resized_array = np.array(resized_image)
    print(f"[{datetime.now()}] 리사이징 후 이미지 배열 형태: {resized_array.shape}, 타입: {resized_array.dtype}")
    
    # 배경 체크 (첫 번째 픽셀 값 출력)
    try:
        print(f"[{datetime.now()}] 리사이징 후 첫 번째 픽셀 값: {resized_array[0, 0]}")
    except:
        print(f"[{datetime.now()}] 픽셀 값 확인 실패")
    
    # 확실하게 RGB 모드로 변환
    if resized_image.mode != 'RGB':
        resized_image = resized_image.convert('RGB')
    
    return resized_image

def should_resize_image(image):
    """이미지 크기 확인 및 리사이징 결정 함수"""
    if image is None:
        return False
        
    width, height = image.size
    # 이미지가 1920x1080 이상이면 리사이징
    return width >= 1920 or height >= 1080

def process_image(img):
    """이미지 전처리 함수 - 기본 처리만 수행"""
    if img is None:
        return None
    
    # PIL Image인지 확인
    if not isinstance(img, Image.Image):
        return img
    
    print(f"[{datetime.now()}] 원본 이미지 모드: {img.mode}, 크기: {img.size}")
    
    # RGBA 이미지는 알파 채널 처리
    if img.mode == 'RGBA':
        # RGB로 변환 (배경은 흰색)
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    # RGB가 아닌 다른 모드는 RGB로 변환
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    # 필요한 경우 리사이징
    if should_resize_image(img):
        img = resize_image(img)
    
    # 최종 결과 로깅
    print(f"[{datetime.now()}] 최종 처리 이미지 모드: {img.mode}")
    
    return img
