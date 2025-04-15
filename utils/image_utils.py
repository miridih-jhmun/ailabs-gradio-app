import os
from PIL import Image

# 기본 이미지 경로 설정 
TEMPLATE_IMAGE_PATH = "/workspace/gradio_app/input/template/001.png"
ELEMENT_IMAGE_PATH = "/workspace/gradio_app/input/element/725494_0_0.png"

def resize_image_by_half(image):
    if image is None:
        return None
        
    # 원본 이미지 크기 가져오기
    width, height = image.size
    
    # 절반 크기로 조정 (비율 유지)
    new_width = width // 3
    new_height = height // 3
    
    # 이미지 리사이즈
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    return resized_image

# 기본 이미지 로드 함수
def get_template_image(resize_half=False):
    if os.path.exists(TEMPLATE_IMAGE_PATH):
        image = Image.open(TEMPLATE_IMAGE_PATH)
        if resize_half:
            return resize_image_by_half(image)
        return image
    return None

def get_element_image(resize_half=False):
    if os.path.exists(ELEMENT_IMAGE_PATH):
        image = Image.open(ELEMENT_IMAGE_PATH)
        if resize_half:
            return resize_image_by_half(image)
        return image
    return None