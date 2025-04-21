import gradio as gr
import numpy as np
from PIL import Image
from inference.quality_models import check_model_status

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
    .error-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ffeeba;
        margin: 10px 0;
        font-weight: bold;
    }
    </style>
    """)
    
    return input_image, style_html

# 오류 메시지 HTML 생성 함수
def create_error_html(error_msg):
    """에러 메시지를 HTML로 포맷팅"""
    if error_msg:
        return f"""
        <div class="error-message">
            {error_msg}
        </div>
        """
    else:
        return ""

# 모델 출력 후처리 함수 - 여러 이미지 처리
def process_output_images(images):
    if images is None:
        return None
    
    processed_images = []
    for img in images:
        processed_images.append(img)
    
    return processed_images

# UI 구성 함수 - 모델 유형에 따라 다른 인터페이스 생성
# is_template: 입력 이미지가 template이면 True, 요소이면 False
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
        
        # 에러 메시지 표시 컴포넌트
        error_html = gr.HTML("")
        
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
            try:
                # 이미지는 그대로 전달
                result = inference_fn(image, text)
                
                # 결과가 tuple이고 길이가 3이면 에러 메시지 포함
                if isinstance(result, tuple) and len(result) == 3:
                    output_images, bm25, style = result[:2]
                    error_msg = result[2]
                    return output_images, bm25, style, create_error_html(error_msg)
                else:
                    output_images, bm25, style = result
                    return output_images, bm25, style, ""
            except Exception as e:
                # 예상치 못한 오류 처리
                error_msg = f"⚠️ 처리 중 오류 발생: {str(e)}"
                return [], 0, 0, create_error_html(error_msg)
        
        submit_btn.click(
            fn=model_wrapper,
            inputs=[input_image, input_text],
            outputs=[output_gallery, bm25_score, style_score, error_html]
        )

def create_model_interface_type2(inference_fn, tab_name, score_label="유사도 점수", is_template=True):
    # 스타일/내용/레이아웃이 비슷한 템플릿 찾기 (이미지 → 이미지들, 점수)
    with gr.Column():
        # 공통 이미지 입력 컴포넌트 사용
        input_image, style_html = create_common_image_input(
            label="입력 이미지 (템플릿)" if is_template else "입력 이미지 (요소)"
        )
        
        submit_btn = gr.Button(f"{tab_name} 실행")
        
        # 에러 메시지 표시 컴포넌트
        error_html = gr.HTML("")
        
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
            try:
                # 모델 실행
                result = inference_fn(image)
                
                # 결과가 tuple이고 길이가 3이면 에러 메시지 포함
                if isinstance(result, tuple) and len(result) == 3:
                    output_images, score = result[:2]
                    error_msg = result[2]
                    return output_images, score, create_error_html(error_msg)
                else:
                    output_images, score = result
                    return output_images, score, ""
            except Exception as e:
                # 예상치 못한 오류 처리
                error_msg = f"⚠️ 처리 중 오류 발생: {str(e)}"
                return [], 0, create_error_html(error_msg)
        
        submit_btn.click(
            fn=model_wrapper,
            inputs=[input_image],
            outputs=[output_gallery, similarity_score, error_html]
        )

def create_model_interface_type3(inference_fn, tab_name, score_label="품질 점수", is_template=False):
    # 저퀄리티 분류 모델 (이미지 → 점수, 라벨)
    with gr.Column():
        # 공통 이미지 입력 컴포넌트 사용
        input_image, style_html = create_common_image_input(
            label="입력 이미지 (템플릿)" if is_template else "입력 이미지 (요소)"
        )
        
        with gr.Row():
            # 모델 로드 버튼과 상태 표시 추가
            load_model_btn = gr.Button("모델 로드", variant="primary")
            model_status = gr.Textbox(label="모델 상태", value="모델이 로드되지 않았습니다.", interactive=False)
        
        # 모델 로드 버튼 클릭 이벤트 처리
        def load_model_fn():
            from inference.quality_models import load_model
            success, message = load_model()
            status_msg = f"✅ {message}" if success else f"❌ {message}"
            return status_msg
            
        # 모델 상태 확인 함수
        def check_model_status_fn():
            loaded, message = check_model_status()
            status_msg = f"✅ {message}" if loaded else f"❌ {message}"
            return status_msg
        
        # 페이지 로드 시 모델 상태 확인
        model_status.value = check_model_status_fn()
        
        # 모델 로드 버튼 클릭 이벤트 연결
        load_model_btn.click(
            fn=load_model_fn,
            inputs=[],
            outputs=[model_status]
        )
        
        submit_btn = gr.Button(f"{tab_name} 실행")
        
        # 에러 메시지 표시 컴포넌트
        error_html = gr.HTML("")
        
        with gr.Row():
            quality_score = gr.Number(label=score_label)
            result_label = gr.Label(label="분류 결과")
        
        # 모델 결과 처리를 위한 래퍼 함수
        def model_wrapper(image):
            try:
                # 모델 실행 전 상태 확인
                loaded, message = check_model_status()
                if not loaded:
                    return 0, "모델 미로드", create_error_html("⚠️ 모델이 로드되지 않았습니다. '모델 로드' 버튼을 클릭하세요.")
                
                # 모델 실행
                result = inference_fn(image)
                
                # 결과가 tuple이고 길이가 3이면 에러 메시지 포함
                if isinstance(result, tuple) and len(result) == 3:
                    score, label, error_msg = result
                    return score, label, create_error_html(error_msg)
                else:
                    score, label = result
                    return score, label, ""
            except Exception as e:
                # 예상치 못한 오류 처리
                error_msg = f"⚠️ 처리 중 오류 발생: {str(e)}"
                return 0, "오류", create_error_html(error_msg)
        
        submit_btn.click(
            fn=model_wrapper,
            inputs=[input_image],
            outputs=[quality_score, result_label, error_html]
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
        
        # 에러 메시지 표시 컴포넌트
        error_html = gr.HTML("")
        
        output_scores = gr.JSON(label="키워드별 점수")
        
        # 모델 결과 처리를 위한 래퍼 함수
        def model_wrapper(image, keywords):
            try:
                # 모델 실행
                result = inference_fn(image, keywords)
                
                # 결과가 tuple이고 길이가 2이면 에러 메시지 포함
                if isinstance(result, tuple) and len(result) == 2:
                    scores, error_msg = result
                    return scores, create_error_html(error_msg)
                else:
                    return result, ""
            except Exception as e:
                # 예상치 못한 오류 처리
                error_msg = f"⚠️ 처리 중 오류 발생: {str(e)}"
                return {}, create_error_html(error_msg)
        
        submit_btn.click(
            fn=model_wrapper,
            inputs=[input_image, keywords],
            outputs=[output_scores, error_html]
        )