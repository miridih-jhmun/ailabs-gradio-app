import numpy as np

# 비슷한 스타일 찾기 모델
def similar_style_inference(image, text):
    # 출력: 이미지 K개, BM25 점수, 스타일 점수
    k = 5  # 결과 이미지 개수
    output_images = [image] * k  # 예시로 K개의 결과 이미지 반환
    bm25_score = np.random.randint(1, 101)
    style_score = np.random.randint(1, 101)
    return output_images, bm25_score, style_score

# 스타일 찾기 모델
def style_search_inference(image, text):
    # 출력: 이미지 K개, BM25 점수, 스타일 점수
    k = 5  # 결과 이미지 개수
    output_images = [image] * k
    bm25_score = np.random.randint(1, 101)
    style_score = np.random.randint(1, 101)
    return output_images, bm25_score, style_score