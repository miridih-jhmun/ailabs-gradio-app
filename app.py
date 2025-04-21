import gradio as gr

# 상대 경로로 import
from inference.style_models import similar_style_inference, style_search_inference
from inference.template_models import (
    similar_style_template_inference, 
    similar_content_template_inference, 
    similar_layout_template_inference
)
from inference.quality_models import (
    design_hub_low_quality_inference, 
    ai_image_low_quality_inference
)
from inference.keyword_models import element_keyword_extraction

from ui.interfaces import (
    create_model_interface_type1, 
    create_model_interface_type2, 
    create_model_interface_type3, 
    create_model_interface_type4
)

# Gradio 인터페이스 생성
with gr.Blocks() as demo:
    gr.Markdown("# 모델 성능 평가를 위한 gradio 앱")
    
    with gr.Tabs():
        with gr.TabItem("비슷한 스타일 찾기(E2E)"):
            create_model_interface_type1(similar_style_inference, "비슷한 스타일 찾기", is_template=False)
            
        with gr.TabItem("스타일 찾기(T2E)"):
            create_model_interface_type1(style_search_inference, "스타일 찾기", is_template=True)
            
        with gr.TabItem("스타일이 비슷한 템플릿 찾기(T2T)"):
            create_model_interface_type2(similar_style_template_inference, "스타일이 비슷한 템플릿 찾기", "스타일 점수", is_template=True)
            
        with gr.TabItem("내용이 비슷한 템플릿 찾기(T2T)"):
            create_model_interface_type2(similar_content_template_inference, "내용이 비슷한 템플릿 찾기", "내용 유사도 점수", is_template=True)
            
        with gr.TabItem("레이아웃이 비슷한 템플릿 찾기(T2T)"):
            create_model_interface_type2(similar_layout_template_inference, "레이아웃이 비슷한 템플릿 찾기", "레이아웃 유사도 점수", is_template=True)
            
        with gr.TabItem("디자인 허브 측면 저퀄리티 분류 모델"):
            create_model_interface_type3(design_hub_low_quality_inference, "디자인 허브 측면 저퀄리티 분류", "품질 점수", is_template=False)
            
        with gr.TabItem("AI 이미지 측면 저퀄리티 분류 모델"):
            create_model_interface_type3(ai_image_low_quality_inference, "AI 이미지 측면 저퀄리티 분류", "품질 점수", is_template=False)
            
        with gr.TabItem("요소 키워드 추출"):
            create_model_interface_type4(element_keyword_extraction, "요소 키워드 추출", is_template=False)

# 외부 접속을 위한 서버 설정
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)