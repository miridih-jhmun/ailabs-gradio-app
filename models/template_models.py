import numpy as np

# 스타일이 비슷한 템플릿 찾기 모델
def similar_style_template_inference(image, text=None):
    # 출력: 이미지 K개, 스타일 점수
    k = 5
    output_images = [image] * k
    style_score = np.random.randint(1, 101)
    return output_images, style_score

# 내용이 비슷한 템플릿 찾기 모델
def similar_content_template_inference(image, text=None):
    # 출력: 이미지 K개, 내용 유사도 점수
    k = 5
    output_images = [image] * k
    content_score = np.random.randint(1, 101)
    return output_images, content_score

# 레이아웃이 비슷한 템플릿 찾기 모델
def similar_layout_template_inference(image, text=None):
    # 출력: 이미지 K개, 레이아웃 유사도 점수
    k = 5
    output_images = [image] * k
    layout_score = np.random.randint(1, 101)
    return output_images, layout_score