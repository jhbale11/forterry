# KSAT 비문학 문제 생성기 - Langchain 버전

# 필요한 라이브러리 설치
# !pip install langchain langchain_openai langchain_anthropic

# 필요한 패키지 임포트
import os
import json
import random
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Langchain 패키지 임포트
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage

class KSATQuestionGenerator:
    def __init__(self, 
                 openai_api_key: Optional[str] = None, 
                 anthropic_api_key: Optional[str] = None,
                 provider: str = "openai",
                 model: Optional[str] = None):
        """
        수능 국어 비문학 문제 생성기 초기화
        
        Args:
            openai_api_key: OpenAI API 키 (없으면 환경 변수에서 가져옴)
            anthropic_api_key: Anthropic API 키 (없으면 환경 변수에서 가져옴)
            provider: 사용할 LLM 제공자 ("openai" 또는 "anthropic")
            model: 사용할 모델명 (None이면 기본값 사용)
        """
        self.provider = provider.lower()
        
        # API 키 설정
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        # 모델 설정
        if self.provider == "openai":
            if self.openai_api_key is None:
                raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
                
            default_model = "gpt-4o"
            self.model_name = model or default_model
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=0.2,
                openai_api_key=self.openai_api_key,
                max_tokens=4000
            )
        elif self.provider == "anthropic":
            if self.anthropic_api_key is None:
                raise ValueError("Anthropic API 키가 설정되지 않았습니다.")
                
            default_model = "claude-3-7-sonnet-20250219"
            self.model_name = model or default_model
            self.llm = ChatAnthropic(
                model=self.model_name,
                temperature=0.2,
                anthropic_api_key=self.anthropic_api_key,
                max_tokens=4000
            )
        else:
            raise ValueError("지원되지 않는 제공자입니다. 'openai' 또는 'anthropic'을 사용하세요.")
        
        # 지문 유형 정보
        self.passage_types = ["인문", "예술", "사회", "과학", "기술"]
        
        # 문제 유형 정보
        self.question_types = {
            "사실적 이해": "지문에 제시된 정보에 대한 직접적인 이해를 묻는 문제",
            "추론적 이해": "지문에 명시적으로 드러나지 않았으나 추론할 수 있는 내용을 묻는 문제",
            "비판적 이해": "지문의 주장이나 설명 방식 등에 대한 평가를 묻는 문제",
            "구체적 적용": "지문의 개념이나 원리를 구체적 상황에 적용하는 문제",
            "보기 이해": "지문과 보기를 연관지어 이해하는 문제",
            "개념 관계": "지문 속 개념들 간의 관계를 파악하는 문제",
            "선지 선택": "보기의 여러 항목 중 적절한 것을 고르는 문제",
            "어휘 교체": "지문 속 어휘를 적절한 다른 표현으로 바꾸는 문제",
            "메모 평가": "지문에 대한 메모 중 적절하지 않은 것을 찾는 문제"
        }
        
        # 문제 타입별 문장 패턴
        self.question_patterns = {
            "사실적 이해": [
                "윗글을 통해 답을 찾을 수 없는 질문은?",
                "윗글의 내용과 일치하지 않는 것은?", 
                "윗글을 이해한 내용으로 적절하지 않은 것은?",
                "(지문 속 중요 개념)에 대한 설명으로 적절하지 않은 것은?",
                "윗글에 대한 이해로 가장 적절한 것은?"
            ],
            "추론적 이해": [
                "윗글을 바탕으로 추론한 내용으로 적절하지 않은 것은?",
                "윗글을 읽고 추론한 내용으로 가장 적절한 것은?",
                "윗글의 내용을 바탕으로 알 수 있는 것은?"
            ],
            "비판적 이해": [
                "윗글에서 필자의 주장을 뒷받침하기 위해 활용한 방안에 대한 비판적 이해로 적절하지 않은 것은?",
                "윗글의 서술 방식에 대한 설명으로 가장 적절한 것은?",
                "윗글의 내용 전개 방식으로 가장 적절한 것은?"
            ],
            "구체적 적용": [
                "다음 상황에 윗글의 내용을 적용한 것으로 가장 적절한 것은?",
                "(지문 속 개념)의 원리가 적용된 사례로 가장 적절한 것은?",
                "(지문 속 상황)과 유사한 원리가 적용된 사례로 가장 적절한 것은?"
            ],
            "보기 이해": [
                "윗글을 참고할 때, <보기>에 대한 반응으로 가장 적절한 것은?",
                "윗글을 바탕으로 <보기>를 이해한 내용으로 가장 적절한 것은?",
                "윗글을 바탕으로 <보기>를 평가한 내용으로 적절하지 않은 것은?"
            ],
            "개념 관계": [
                "(개념 A)와/과 (개념 B)의 관계에 대한 설명으로 적절한 것은?",
                "윗글에 제시된 (개념 A)와/과 (개념 B)에 대한 설명으로 가장 적절한 것은?"
            ],
            "선지 선택": [
                "윗글을 읽고 해결할 수 있는 질문으로 적절한 것을 <보기>에서 모두 고른 것은?",
                "<보기>의 견해와 윗글의 견해가 일치하는 것을 고른 것은?"
            ],
            "어휘 교체": [
                "문맥상 ⓐ ~ ⓔ와 바꿔 쓰기에 가장 적절한 것은?",
                "밑줄 친 (단어)의 문맥적 의미와 가장 가까운 것은?"
            ],
            "메모 평가": [
                "윗글을 학습하면서 작성한 다음 학생의 메모 내용 중 적절하지 않은 것은?"
            ]
        }
        
    def set_api_key(self, api_key: str, provider: str = "openai") -> None:
        """API 키 설정"""
        if provider.lower() == "openai":
            self.openai_api_key = api_key
            os.environ["OPENAI_API_KEY"] = api_key
            
            # ChatOpenAI 모델 재초기화
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=0.2,
                openai_api_key=self.openai_api_key,
                max_tokens=4000
            )
            self.provider = "openai"
        elif provider.lower() == "anthropic":
            self.anthropic_api_key = api_key
            os.environ["ANTHROPIC_API_KEY"] = api_key
            
            # ChatAnthropic 모델 재초기화
            self.llm = ChatAnthropic(
                model=self.model_name,
                temperature=0.2,
                anthropic_api_key=self.anthropic_api_key,
                max_tokens=4000
            )
            self.provider = "anthropic"
        else:
            raise ValueError("지원되지 않는 제공자입니다. 'openai' 또는 'anthropic'을 사용하세요.")
    
    def analyze_passage(self, passage: str) -> Dict[str, Any]:
        """
        지문을 분석하여 유형과 적합한 문제 유형 결정
        
        Args:
            passage: 지문 텍스트
            
        Returns:
            Dict: 분석 결과 (passage_type, recommended_question_types, keywords 등)
        """
        system_message = SystemMessage(
            content="당신은 수능 국어 비문학 지문 분석 전문가입니다. 지문의 내용, 주요 개념, 구조를 정확히 파악하여 분석해주세요."
        )
        
        prompt = f"""
        다음 수능 국어 비문학 지문을 분석해주세요:
        
        {passage}
        
        분석 시 다음 사항을 포함해주세요:
        1. 지문 유형: 인문, 예술, 사회, 과학, 기술 중 하나
        2. 지문의 핵심 주제
        3. 지문에 등장하는 중요 개념과 용어 (최소 3개)
        4. 지문의 주요 논지나 내용 전개 방식
        5. 이 지문에 적합한 문제 유형 (최소 3가지):
           - 사실적 이해 (지문에 제시된 정보에 대한 직접적인 이해)
           - 추론적 이해 (지문에서 추론할 수 있는 내용)
           - 비판적 이해 (지문의 주장이나 설명 방식 등에 대한 평가)
           - 구체적 적용 (지문의 개념이나 원리를 구체적 상황에 적용)
           - 보기 이해 (지문과 보기를 연관지어 이해)
           - 개념 관계 (지문 속 개념들 간의 관계 파악)
           - 선지 선택 (보기의 여러 항목 중 적절한 것 고르기)
           - 어휘 교체 (지문 속 어휘를 적절한 다른 표현으로 바꾸기)
           - 메모 평가 (지문에 대한 메모 중 적절하지 않은 것 찾기)
        
        분석 결과는 다음 형식으로 작성해주세요:
        {{
            "passage_type": "지문 유형",
            "theme": "핵심 주제",
            "key_concepts": ["개념1", "개념2", "개념3", ...],
            "structure": "내용 전개 방식",
            "recommended_question_types": ["유형1", "유형2", "유형3", ...],
            "rationale": "문제 유형 추천 이유"
        }}
        """
        
        human_message = HumanMessage(content=prompt)
        
        try:
            response = self.llm([system_message, human_message])
            analysis_text = response.content
            
            # JSON 문자열을 파싱하여 딕셔너리로 변환
            try:
                start_idx = analysis_text.find('{')
                end_idx = analysis_text.rfind('}') + 1
                json_str = analysis_text[start_idx:end_idx]
                analysis = json.loads(json_str)
            except:
                # JSON 파싱에 실패한 경우, 전체 응답을 분석 결과로 사용
                analysis = {
                    "passage_type": "사회",  # 기본값
                    "theme": "파악 불가",
                    "key_concepts": [],
                    "structure": "파악 불가",
                    "recommended_question_types": ["사실적 이해", "추론적 이해", "보기 이해"],
                    "rationale": "자동 분석 실패",
                    "raw_response": analysis_text
                }
            
            return analysis
                
        except Exception as e:
            print(f"지문 분석 중 오류 발생: {e}")
            return {
                "passage_type": "사회",  # 기본값
                "theme": "분석 오류",
                "key_concepts": [],
                "structure": "분석 오류",
                "recommended_question_types": ["사실적 이해", "추론적 이해", "보기 이해"],
                "rationale": f"분석 중 오류가 발생했습니다: {str(e)}"
            }
    
    def generate_question(self, 
                        passage: str, 
                        question_type: str, 
                        analysis: Dict[str, Any] = None,
                        include_negation: bool = None, 
                        specific_pattern: str = None) -> str:
        """
        지문과 문제 유형을 기반으로 KSAT 문제 생성
        
        Args:
            passage: 지문 텍스트
            question_type: 문제 유형 (사실적 이해, 추론적 이해 등)
            analysis: 지문 분석 결과 (없으면 자동 분석)
            include_negation: 부정형 문제 여부 (None이면 자동 결정)
            specific_pattern: 특정 문제 패턴 (None이면 자동 선택)
            
        Returns:
            str: 생성된 문제
        """
        # 분석 결과가 없으면 지문 자동 분석
        if analysis is None:
            analysis = self.analyze_passage(passage)
        
        # 문제 패턴 선택
        if specific_pattern:
            question_pattern = specific_pattern
        elif question_type in self.question_patterns:
            # 50%는 부정형, 50%는 긍정형 문제 생성
            if include_negation is None:
                include_negation = random.random() < 0.5
                
            # 부정형/긍정형에 맞는 패턴 필터링
            filtered_patterns = []
            for pattern in self.question_patterns[question_type]:
                is_negation_pattern = "않은" in pattern or "없는" in pattern
                if include_negation == is_negation_pattern:
                    filtered_patterns.append(pattern)
            
            # 필터링된 패턴이 없으면 모든 패턴 사용
            if not filtered_patterns:
                filtered_patterns = self.question_patterns[question_type]
                
            question_pattern = random.choice(filtered_patterns)
        else:
            question_pattern = "윗글에 대한 이해로 가장 적절한 것은?"
        
        # 문제에 보기가 필요한지 확인
        needs_example = question_type in ["보기 이해", "선지 선택"]
        
        # 문제 작성에 필요한 보기 관련 텍스트를 미리 준비 (백슬래시 문제 회피)
        example_instruction = "<보기>를 포함하여 " if needs_example else ""
        example_format = "[보기]\n(적절한 보기 내용 - 지문과 연관된 새로운 사례, 대화, 예시 등)" if needs_example else ""
        
        system_message = SystemMessage(
            content="당신은 수능 국어 비문학 문제 출제 전문가입니다. 주어진 지문을 분석하고 수능 형식에 맞는 양질의 문제를 생성해주세요."
        )
        
        prompt = f"""
        다음 지문을 바탕으로 "{question_type}" 유형의 수능 국어 비문학 문제를 생성해주세요.
        
        지문:
        {passage}
        
        문제 생성 지침:
        1. 문제 유형: {question_type}
        2. 질문 형식: "{question_pattern}"
        3. 객관식 5지선다형으로 작성
        4. {example_instruction}지문의 내용을 정확히 이해해야 풀 수 있는 문제 생성
        5. 모든 선택지는 지문과 연관된 내용으로 작성
        6. 정답은 반드시 지문에서 근거를 찾을 수 있어야 함
        7. 오답은 그럴듯하게 작성하되, 확실히 틀린 내용이어야 함
        
        지문 분석 정보:
        - 지문 유형: {analysis['passage_type']}
        - 핵심 주제: {analysis['theme']}
        - 주요 개념: {', '.join(analysis['key_concepts'])}
        - 내용 전개 방식: {analysis['structure']}
        
        문제 작성 형식:
        [문제]
        {question_pattern}
        
        {example_format}
        
        [선택지]
        ① (선택지 1)
        ② (선택지 2)
        ③ (선택지 3)
        ④ (선택지 4)
        ⑤ (선택지 5)
        
        [정답]
        (정답 번호)
        
        [해설]
        (정답 선택지에 대한 해설 및 나머지 선택지가 오답인 이유)
        """
        
        human_message = HumanMessage(content=prompt)
        
        try:
            response = self.llm([system_message, human_message])
            return response.content
        
        except Exception as e:
            print(f"문제 생성 중 오류 발생: {e}")
            return f"문제 생성 중 오류가 발생했습니다: {str(e)}"
    
    def generate_question_set(self, passage: str, num_questions: int = 3) -> Dict[str, Any]:
        """
        지문에 대한 여러 유형의 문제 세트 생성
        
        Args:
            passage: 지문 텍스트
            num_questions: 생성할 문제 수
            
        Returns:
            Dict: 분석 결과와 생성된 문제들
        """
        # 지문 분석
        analysis = self.analyze_passage(passage)
        
        results = {
            "analysis": analysis,
            "questions": []
        }
        
        # 분석 결과에서 추천된 문제 유형 사용
        question_types = analysis.get("recommended_question_types", ["사실적 이해", "추론적 이해", "보기 이해"])
        
        # 문제 수가 추천된 유형보다 많으면 중복 허용
        if num_questions > len(question_types):
            while len(question_types) < num_questions:
                question_types.append(random.choice(list(self.question_types.keys())))
        
        # 문제 생성
        for i in range(min(num_questions, len(question_types))):
            question_type = question_types[i]
            
            # 부정형과 긍정형 문제 번갈아 생성
            include_negation = i % 2 == 0
            
            question = self.generate_question(
                passage=passage,
                question_type=question_type,
                analysis=analysis,
                include_negation=include_negation
            )
            
            results["questions"].append({
                "question_type": question_type,
                "content": question
            })
        
        return results
        
    def save_to_file(self, results: Dict[str, Any], filename: str = None) -> str:
        """
        결과를 파일로 저장
        
        Args:
            results: 생성 결과
            filename: 파일명 (None이면 자동 생성)
            
        Returns:
            str: 저장된 파일 경로
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ksat_questions_{timestamp}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            # 분석 결과 저장
            f.write("=== 지문 분석 결과 ===\n")
            f.write(f"지문 유형: {results['analysis']['passage_type']}\n")
            f.write(f"핵심 주제: {results['analysis']['theme']}\n")
            f.write(f"주요 개념: {', '.join(results['analysis']['key_concepts'])}\n")
            f.write(f"내용 전개 방식: {results['analysis']['structure']}\n")
            f.write(f"추천 문제 유형: {', '.join(results['analysis']['recommended_question_types'])}\n")
            if 'rationale' in results['analysis']:
                f.write(f"추천 이유: {results['analysis']['rationale']}\n")
            f.write("\n")
            
            # 각 문제 저장
            for i, question_data in enumerate(results["questions"]):
                f.write(f"=== 문제 {i+1} ({question_data['question_type']}) ===\n")
                f.write(f"{question_data['content']}\n\n")
        
        return filename

def mark_key_vocabulary(passage: str, num_words: int = 5) -> str:
    """
    지문에서 어휘 교체 문제를 위한 키워드 마킹
    
    Args:
        passage: 원본 지문
        num_words: 마킹할 단어 수
        
    Returns:
        str: 키워드가 마킹된 지문
    """
    # 단어 추출 (2자 이상의 한글 단어)
    words = re.findall(r'[가-힣]{2,}', passage)
    word_counts = {}
    
    # 단어 빈도 계산
    for word in words:
        if word not in word_counts:
            word_counts[word] = 0
        word_counts[word] += 1
    
    # 빈도가 높은 단어들 중에서 선택 (2번 이상 등장하는 단어 우선)
    frequent_words = sorted([(word, count) for word, count in word_counts.items() if count >= 2], 
                           key=lambda x: x[1], reverse=True)
    
    # 빈도가 높은 단어가 충분하지 않으면 다른 단어도 포함
    if len(frequent_words) < num_words:
        other_words = sorted([(word, count) for word, count in word_counts.items() if count == 1], 
                            key=lambda x: len(x[0]), reverse=True)
        frequent_words.extend(other_words)
    
    # 최종 선택된 단어들
    selected_words = [word for word, _ in frequent_words[:num_words]]
    
    # 선택된 단어를 ⓐ, ⓑ, ⓒ, ⓓ, ⓔ로 마킹
    markers = ['ⓐ', 'ⓑ', 'ⓒ', 'ⓓ', 'ⓔ']
    marked_passage = passage
    
    for i, word in enumerate(selected_words):
        if i < len(markers):
            # 단어의 첫 등장만 마킹 (단어 경계 고려)
            # 정규표현식 패턴을 직접 f-string 내에서 사용하지 않고 별도로 구성
            pattern = r'(?<!\w)' + re.escape(word) + r'(?!\w)'
            marked_passage = re.sub(pattern, markers[i] + ' ' + word, marked_passage, count=1)
    
    return marked_passage

def read_passage_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        passage = f.read()
    return passage