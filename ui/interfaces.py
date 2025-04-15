import gradio as gr
import numpy as np
from utils.image_utils import get_template_image, get_element_image
from PIL import Image

# 이미지 크기 확인 및 리사이징 결정 함수 추가
def should_resize_image(image):
    if image is None:
        return False
        
    width, height = image.size
    # 이미지가 1920x1080 이상이면 리사이징
    return width >= 1920 or height >= 1080

# PNG 이미지 처리 함수 - 알파 채널 처리
def process_image_for_display(img):
    if img is None:
        return None
    
    # PIL Image인지 확인
    if not isinstance(img, Image.Image):
        return img
    
    # PNG 이미지의 알파 채널 처리
    if img.mode == 'RGBA':
        # 알파 채널이 있는 이미지를 RGB로 변환 (흰색 배경 사용)
        background = Image.new('RGBA', img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img).convert('RGB')
    
    return img

# 모델 출력 후처리 함수 - 여러 이미지 처리
def process_output_images(images):
    if images is None:
        return None
    
    processed_images = []
    for img in images:
        processed_img = process_image_for_display(img)
        processed_images.append(processed_img)
    
    return processed_images

# UI 구성 함수 - 모델 유형에 따라 다른 인터페이스 생성
def create_model_interface_type1(inference_fn, tab_name, use_template=False):
    # 비슷한 스타일 찾기, 스타일 찾기 (이미지, 텍스트 → 이미지들, BM25점수, 스타일점수)
    with gr.Column():
        # 이미지 가져오기
        raw_img = get_template_image() if use_template else get_element_image()
        # 이미지 크기 확인 후 필요시 리사이징
        resize_half = should_resize_image(raw_img)
        default_img = get_template_image(resize_half=resize_half) if use_template else get_element_image(resize_half=resize_half)
        
        # 입력 이미지 처리
        default_img = process_image_for_display(default_img)
        
        # 입력 컴포넌트
        input_image = gr.Image(type="pil", label="입력 이미지", value=default_img, show_label=True, show_download_button=True, scale=1, container=True)
        input_text = gr.Textbox(label="입력 텍스트", value="")
        
        # 버튼 생성 - 항상 활성화 상태
        submit_btn = gr.Button(f"{tab_name} 실행")
        
        # 갤러리 설정 개선
        output_gallery = gr.Gallery(
            label="출력 이미지", 
            columns=3, 
            show_label=True, 
            object_fit="none",
            height="auto",
            container=False
        )
        
        with gr.Row():
            bm25_score = gr.Number(label="BM25 점수")
            style_score = gr.Number(label="스타일 점수")
        
        # 모델 결과 처리를 위한 래퍼 함수
        def model_wrapper(image, text):
            # 모델 실행
            output_images, bm25, style = inference_fn(image, text)
            # 출력 이미지 처리
            processed_images = process_output_images(output_images)
            return processed_images, bm25, style
        
        submit_btn.click(
            fn=model_wrapper,
            inputs=[input_image, input_text],
            outputs=[output_gallery, bm25_score, style_score]
        )

def create_model_interface_type2(inference_fn, tab_name, score_label="유사도 점수", use_template=True):
    # 스타일/내용/레이아웃이 비슷한 템플릿 찾기 (이미지 → 이미지들, 점수)
    with gr.Column():
        # 이미지 가져오기
        raw_img = get_template_image() if use_template else get_element_image()
        # 이미지 크기 확인 후 필요시 리사이징
        resize_half = should_resize_image(raw_img)
        default_img = get_template_image(resize_half=resize_half) if use_template else get_element_image(resize_half=resize_half)
        
        # 입력 이미지 처리
        default_img = process_image_for_display(default_img)
        
        input_image = gr.Image(type="pil", label="입력 이미지", value=default_img, show_label=True, show_download_button=True, scale=1, container=True)
        submit_btn = gr.Button(f"{tab_name} 실행")
        
        # 갤러리 설정 개선
        output_gallery = gr.Gallery(
            label="출력 이미지", 
            columns=3, 
            show_label=True, 
            object_fit="contain",
            height="auto",
            container=True
        )
        
        similarity_score = gr.Number(label=score_label)
        
        # 모델 결과 처리를 위한 래퍼 함수
        def model_wrapper(image):
            # 모델 실행
            output_images, score = inference_fn(image)
            # 출력 이미지 처리
            processed_images = process_output_images(output_images)
            return processed_images, score
        
        submit_btn.click(
            fn=model_wrapper,
            inputs=[input_image],
            outputs=[output_gallery, similarity_score]
        )

def create_model_interface_type3(inference_fn, tab_name, score_label="품질 점수", use_template=False):
    # 저퀄리티 분류 모델 (이미지 → 점수, 라벨)
    with gr.Column():
        # 이미지 가져오기
        raw_img = get_template_image() if use_template else get_element_image()
        # 이미지 크기 확인 후 필요시 리사이징
        resize_half = should_resize_image(raw_img)
        default_img = get_template_image(resize_half=resize_half) if use_template else get_element_image(resize_half=resize_half)
        
        # 입력 이미지 처리
        default_img = process_image_for_display(default_img)
        
        input_image = gr.Image(type="pil", label="입력 이미지", value=default_img, show_label=True, show_download_button=True, scale=1, container=True)
        submit_btn = gr.Button(f"{tab_name} 실행")
        
        with gr.Row():
            quality_score = gr.Number(label=score_label)
            result_label = gr.Label(label="분류 결과")
        
        submit_btn.click(
            fn=inference_fn,
            inputs=[input_image],
            outputs=[quality_score, result_label]
        )

def create_model_interface_type4(inference_fn, tab_name, use_template=False):
    # 요소 키워드 추출 (이미지, 키워드 리스트 → 키워드별 점수)
    with gr.Column():
        # 이미지 가져오기
        raw_img = get_template_image() if use_template else get_element_image()
        # 이미지 크기 확인 후 필요시 리사이징
        resize_half = should_resize_image(raw_img)
        default_img = get_template_image(resize_half=resize_half) if use_template else get_element_image(resize_half=resize_half)
        
        # 입력 이미지 처리
        default_img = process_image_for_display(default_img)
        
        input_image = gr.Image(type="pil", label="입력 이미지", value=default_img, show_label=True, show_download_button=True, scale=1, container=True)
        keywords = gr.Textbox(label="키워드 리스트 (쉼표로 구분)", value="디자인,로고,아이콘,배너")
        submit_btn = gr.Button(f"{tab_name} 실행")
        
        output_scores = gr.JSON(label="키워드별 점수")
        
        submit_btn.click(
            fn=inference_fn,
            inputs=[input_image, keywords],
            outputs=[output_scores]
        )