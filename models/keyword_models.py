import numpy as np

# 요소 키워드 추출
def element_keyword_extraction(image, keywords):
    # 입력: 이미지, 키워드 리스트
    # 출력: 키워드별 점수
    keyword_list = keywords.split(',')
    scores = {keyword.strip(): np.random.randint(1, 101) for keyword in keyword_list}
    return scores