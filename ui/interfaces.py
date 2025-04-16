import gradio as gr
import numpy as np
from PIL import Image

# 공통 이미지 입력 컴포넌트 생성 함수
def create_common_image_input(label="입력 이미지"):
    """모든 인터페이스에 공통적으로 사용되는 이미지 입력 컴포넌트 생성"""
    # 이미지 입력 컴포넌트
    input_image = gr.Image(
        type="pil",
        label=label,
        width="100%",                   # 너비는 100%
        height="auto",                  # 높이는 자동 계산
        scale=1,
        show_label=True,
        show_download_button=True,
        container=True,
        interactive=True,
        image_mode="RGB",
        sources=["upload", "clipboard"],
        elem_classes="max-height-image"  # 클래스 추가
    )
    
    # CSS 스타일 추가
    style_html = gr.HTML("""
    <style>
    .max-height-image img {
        max-height: 400px !important;
        object-fit: contain !important;
    }
    </style>
    """)
    
    return input_image, style_html

# 모델 출력 후처리 함수 - 여러 이미지 처리
def process_output_images(images):
    if images is None:
        return None
    
    processed_images = []
    for img in images:
        processed_images.append(img)
    
    return processed_images

# UI 구성 함수 - 모델 유형에 따라 다른 인터페이스 생성
def create_model_interface_type1(inference_fn, tab_name, is_template=False):
    # 비슷한 스타일 찾기, 스타일 찾기 (이미지, 텍스트 → 이미지들, BM25점수, 스타일점수)
    with gr.Column():
        # 공통 이미지 입력 컴포넌트 사용
        input_image, style_html = create_common_image_input(
            label="입력 이미지 (템플릿)" if is_template else "입력 이미지 (요소)"
        )
        
        input_text = gr.Textbox(label="입력 텍스트", value="")
        
        # 버튼 생성 - 항상 활성화 상태
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
        
        with gr.Row():
            bm25_score = gr.Number(label="BM25 점수")
            style_score = gr.Number(label="스타일 점수")
        
        # 모델 결과 처리를 위한 래퍼 함수
        def model_wrapper(image, text):
            # 이미지는 그대로 전달
            output_images, bm25, style = inference_fn(image, text)
            return output_images, bm25, style
        
        submit_btn.click(
            fn=model_wrapper,
            inputs=[input_image, input_text],
            outputs=[output_gallery, bm25_score, style_score]
        )

def create_model_interface_type2(inference_fn, tab_name, score_label="유사도 점수", is_template=True):
    # 스타일/내용/레이아웃이 비슷한 템플릿 찾기 (이미지 → 이미지들, 점수)
    with gr.Column():
        # 공통 이미지 입력 컴포넌트 사용
        input_image, style_html = create_common_image_input(
            label="입력 이미지 (템플릿)" if is_template else "입력 이미지 (요소)"
        )
        
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
            return output_images, score
        
        submit_btn.click(
            fn=model_wrapper,
            inputs=[input_image],
            outputs=[output_gallery, similarity_score]
        )

def create_model_interface_type3(inference_fn, tab_name, score_label="품질 점수", is_template=False):
    # 저퀄리티 분류 모델 (이미지 → 점수, 라벨)
    with gr.Column():
        # 공통 이미지 입력 컴포넌트 사용
        input_image, style_html = create_common_image_input(
            label="입력 이미지 (템플릿)" if is_template else "입력 이미지 (요소)"
        )
        
        submit_btn = gr.Button(f"{tab_name} 실행")
        
        with gr.Row():
            quality_score = gr.Number(label=score_label)
            result_label = gr.Label(label="분류 결과")
        
        # 모델 결과 처리를 위한 래퍼 함수
        def model_wrapper(image):
            # 모델 실행
            score, label = inference_fn(image)
            return score, label
        
        submit_btn.click(
            fn=model_wrapper,
            inputs=[input_image],
            outputs=[quality_score, result_label]
        )

def create_model_interface_type4(inference_fn, tab_name, is_template=False):
    # 요소 키워드 추출 (이미지, 키워드 리스트 → 키워드별 점수)
    with gr.Column():
        # 공통 이미지 입력 컴포넌트 사용
        input_image, style_html = create_common_image_input(
            label="입력 이미지 (템플릿)" if is_template else "입력 이미지 (요소)"
        )
        
        keywords = gr.Textbox(label="키워드 리스트 (쉼표로 구분)", value="디자인,로고,아이콘,배너")
        submit_btn = gr.Button(f"{tab_name} 실행")
        
        output_scores = gr.JSON(label="키워드별 점수")
        
        # 모델 결과 처리를 위한 래퍼 함수
        def model_wrapper(image, keywords):
            # 모델 실행
            scores = inference_fn(image, keywords)
            return scores
        
        submit_btn.click(
            fn=model_wrapper,
            inputs=[input_image, keywords],
            outputs=[output_scores]
        )