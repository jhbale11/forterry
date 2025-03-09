# KSAT 문학 문제 생성기 - Langchain 버전

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

class KSATLiteratureQuestionGenerator:
    def __init__(self, 
                 openai_api_key: Optional[str] = None, 
                 anthropic_api_key: Optional[str] = None,
                 provider: str = "openai",
                 model: Optional[str] = None):
        """
        수능 국어 문학 문제 생성기 초기화
        
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
        
        # 문학 장르 정보
        self.literature_genres = {
            "현대시": "현대 시인의 시적 표현과 주제를 다루는 장르",
            "고전시가": "조선시대 이전의 시가 작품으로 시조, 가사, 향가 등이 포함됨",
            "현대소설": "현대의 서사적 산문 문학으로 인물, 사건, 배경을 통해 삶을 형상화",
            "고전소설": "조선시대 이전의 서사적 산문 문학으로 전통적 세계관이 반영됨",
            "수필": "작가의 개인적 경험과 사색을 자유롭게 표현한 산문 문학",
            "희곡": "무대 공연을 전제로 대사와 지문으로 구성된 극 문학",
            "시나리오": "영화 제작을 위한 대본으로 대사와 장면 지시문으로 구성됨"
        }
        
        # 문제 유형 정보
        self.question_types = {
            "작품 이해": "작품의 내용과 주제를 정확히 파악하는 문제",
            "표현상 특징": "작품의 표현 방식, 기법, 문체적 특징을 파악하는 문제",
            "서술상 특징": "작품의 서술 방식, 시점, 구성 등을 파악하는 문제",
            "작가 의도": "작가의 의도와 관점을 파악하는 문제",
            "인물 분석": "작품 속 인물의 성격, 행동, 심리를 분석하는 문제",
            "갈등 양상": "작품 속 갈등의 양상과 의미를 분석하는 문제",
            "상징과 이미지": "작품 속 상징과 이미지의 의미를 파악하는 문제",
            "시어와 구절": "시어나 특정 구절의 의미와 효과를 파악하는 문제",
            "배경 분석": "작품의 시간적, 공간적 배경의 의미를 분석하는 문제",
            "비교와 대조": "둘 이상의 작품을 비교하고 대조하는 문제",
            "작품 평가": "작품의 가치와 의미를 평가하는 문제",
            "외적 준거": "외부 자료나 관점을 적용하여 작품을 이해하는 문제",
            "화자/서술자": "화자나 서술자의 특성과 역할을 파악하는 문제"
        }
        
        # 문제 타입별 문장 패턴
        self.question_patterns = {
            "작품 이해": [
                "윗글에 대한 이해로 적절하지 않은 것은?",
                "윗글의 내용과 일치하지 않는 것은?",
                "작품의 내용으로 볼 때, 가장 적절한 것은?"
            ],
            "표현상 특징": [
                "윗글에 대한 표현상 특징으로 가장 적절한 것은?",
                "윗글에 활용된 표현 기법으로 적절하지 않은 것은?",
                "다음 작품의 표현상 특징에 대한 설명으로 가장 적절한 것은?"
            ],
            "서술상 특징": [
                "작품의 서술상 특징으로 가장 적절한 것은?",
                "작품의 서술 방식에 대한 설명으로 적절하지 않은 것은?",
                "다음 작품의 구성 방식에 대한 설명으로 가장 적절한 것은?"
            ],
            "작가 의도": [
                "작가가 이 작품을 통해 말하고자 하는 바로 가장 적절한 것은?",
                "작가의 의도로 적절하지 않은 것은?",
                "작가가 이 작품에서 궁극적으로 전달하고자 하는 메시지는?"
            ],
            "인물 분석": [
                "(인물)의 성격과 행동에 대한 설명으로 가장 적절한 것은?",
                "(인물)에 대한 이해로 적절하지 않은 것은?",
                "다음 인물들의 관계에 대한 설명으로 가장 적절한 것은?"
            ],
            "갈등 양상": [
                "작품에 나타난 갈등의 양상으로 가장 적절한 것은?",
                "작품 속 갈등의 원인과 전개 과정에 대한 설명으로 적절하지 않은 것은?",
                "주요 갈등의 해소 방식에 대한 설명으로 가장 적절한 것은?"
            ],
            "상징과 이미지": [
                "작품에 나타난 상징적 의미로 가장 적절한 것은?",
                "(구절/소재)의 상징적 의미로 적절하지 않은 것은?",
                "작품에 사용된 이미지의 효과로 가장 적절한 것은?"
            ],
            "시어와 구절": [
                "밑줄 친 (시어/구절)의 의미로 가장 적절한 것은?",
                "(시어/구절)에 대한 설명으로 적절하지 않은 것은?",
                "(구절)이 갖는 함축적 의미로 가장 적절한 것은?"
            ],
            "배경 분석": [
                "작품의 배경이 지니는 의미로 가장 적절한 것은?",
                "작품의 시간적/공간적 배경에 대한 설명으로 적절하지 않은 것은?",
                "작품 속 공간의 상징적 의미로 가장 적절한 것은?"
            ],
            "비교와 대조": [
                "(가)와 (나)의 공통점으로 가장 적절한 것은?",
                "(가)와 (나)의 차이점에 대한 설명으로 적절하지 않은 것은?",
                "두 작품의 표현 방식을 비교한 내용으로 가장 적절한 것은?"
            ],
            "작품 평가": [
                "작품의 문학사적 의의로 가장 적절한 것은?",
                "이 작품에 대한 평가로 적절하지 않은 것은?",
                "작품의 현대적 의미로 가장 적절한 것은?"
            ],
            "외적 준거": [
                "<보기>를 바탕으로 윗글을 감상한 내용으로 적절하지 않은 것은?",
                "<보기>의 관점에서 작품을 이해한 것으로 가장 적절한 것은?",
                "<보기>의 비평 관점을 적용하여 작품을 감상한 내용으로 적절하지 않은 것은?"
            ],
            "화자/서술자": [
                "작품의 화자(서술자)에 대한 설명으로 가장 적절한 것은?",
                "화자(서술자)의 태도에 대한 설명으로 적절하지 않은 것은?",
                "화자(서술자)의 시점 변화가 작품에 미치는 효과로 가장 적절한 것은?"
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
    
    def analyze_literature(self, text: str, title: str = None, author: str = None) -> Dict[str, Any]:
        """
        문학 작품을 분석하여 장르와 적합한 문제 유형 결정
        
        Args:
            text: 문학 작품 텍스트
            title: 제목 (없으면 None)
            author: 작가 (없으면 None)
            
        Returns:
            Dict: 분석 결과 (genre, theme, literary_devices, characters 등)
        """
        system_message = SystemMessage(
            content="당신은 수능 국어 문학 작품 분석 전문가입니다. 작품의 장르, 주제, 표현 기법, 특징을 정확히 파악하여 분석해주세요."
        )
        
        title_info = f"제목: {title}\n" if title else ""
        author_info = f"작가: {author}\n" if author else ""
        
        prompt = f"""
        다음 문학 작품을 분석해주세요:
        
        {title_info}
        {author_info}
        작품:
        {text}
        
        분석 시 다음 사항을 포함해주세요:
        1. 장르: 현대시, 고전시가, 현대소설, 고전소설, 수필, 희곡, 시나리오 중 하나
        2. 작품의 주제
        3. 작품의 핵심 내용 요약
        4. 주요 인물 (소설, 희곡 등의 경우)
        5. 주요 표현 기법 및 문체적 특징
        6. 작품의 구성 및 서술 방식
        7. 작품에 나타난 의미 있는 소재 및 이미지
        8. 이 작품에 적합한 문제 유형 (최소 4가지):
           - 작품 이해 (작품의 내용과 주제 파악)
           - 표현상 특징 (표현 방식, 기법, 문체적 특징)
           - 서술상 특징 (서술 방식, 시점, 구성 등)
           - 작가 의도 (작가의 의도와 관점)
           - 인물 분석 (인물의 성격, 행동, 심리)
           - 갈등 양상 (갈등의 양상과 의미)
           - 상징과 이미지 (상징과 이미지의 의미)
           - 시어와 구절 (시어나 특정 구절의 의미와 효과)
           - 배경 분석 (시간적, 공간적 배경의 의미)
           - 비교와 대조 (둘 이상의 작품 비교)
           - 작품 평가 (작품의 가치와 의미 평가)
           - 외적 준거 (외부 자료나 관점 적용)
           - 화자/서술자 (화자나 서술자의 특성과 역할)
        
        분석 결과는 다음 형식으로 작성해주세요:
        {{
            "genre": "장르",
            "theme": "주제",
            "summary": "핵심 내용 요약",
            "characters": ["인물1", "인물2", ...],
            "literary_devices": ["기법1", "기법2", ...],
            "narrative_structure": "구성 및 서술 방식",
            "symbols_and_images": ["소재/이미지1", "소재/이미지2", ...],
            "recommended_question_types": ["유형1", "유형2", "유형3", "유형4", ...],
            "question_suggestions": ["구체적인 문제 출제 방향 제안1", "제안2", ...],
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
                    "genre": "현대소설",  # 기본값
                    "theme": "파악 불가",
                    "summary": "파악 불가",
                    "characters": [],
                    "literary_devices": [],
                    "narrative_structure": "파악 불가",
                    "symbols_and_images": [],
                    "recommended_question_types": ["작품 이해", "표현상 특징", "인물 분석", "작가 의도"],
                    "question_suggestions": [],
                    "rationale": "자동 분석 실패",
                    "raw_response": analysis_text
                }
            
            return analysis
                
        except Exception as e:
            print(f"작품 분석 중 오류 발생: {e}")
            return {
                "genre": "현대소설",  # 기본값
                "theme": "분석 오류",
                "summary": "분석 오류",
                "characters": [],
                "literary_devices": [],
                "narrative_structure": "분석 오류",
                "symbols_and_images": [],
                "recommended_question_types": ["작품 이해", "표현상 특징", "인물 분석", "작가 의도"],
                "question_suggestions": [],
                "rationale": f"분석 중 오류가 발생했습니다: {str(e)}"
            }
    
    def generate_question(self, 
                        text: str, 
                        question_type: str, 
                        analysis: Dict[str, Any] = None,
                        include_negation: bool = None, 
                        specific_pattern: str = None,
                        title: str = None,
                        author: str = None,
                        add_explanation: bool = True) -> str:
        """
        문학 작품과 문제 유형을 기반으로 KSAT 문제 생성
        
        Args:
            text: 문학 작품 텍스트
            question_type: 문제 유형 (작품 이해, 표현상 특징 등)
            analysis: 작품 분석 결과 (없으면 자동 분석)
            include_negation: 부정형 문제 여부 (None이면 자동 결정)
            specific_pattern: 특정 문제 패턴 (None이면 자동 선택)
            title: 작품 제목 (없으면 None)
            author: 작가 (없으면 None)
            add_explanation: 해설 추가 여부
            
        Returns:
            str: 생성된 문제
        """
        # 분석 결과가 없으면 작품 자동 분석
        if analysis is None:
            analysis = self.analyze_literature(text, title, author)
        
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
            question_pattern = "작품에 대한 이해로 가장 적절한 것은?"
        
        # 문제에 보기가 필요한지 확인
        needs_example = question_type in ["외적 준거", "비교와 대조"]
        
        # 문제 작성에 필요한 보기 관련 텍스트를 미리 준비 (백슬래시 문제 회피)
        example_instruction = "<보기>를 포함하여 " if needs_example else ""
        example_format = "[보기]\n(적절한 보기 내용 - 작품과 연관된 문학사적 맥락, 작가 정보, 비평 관점 등)" if needs_example else ""
        
        title_info = f"\n제목: {title}" if title else ""
        author_info = f"\n작가: {author}" if author else ""
        
        system_message = SystemMessage(
            content="당신은 수능 국어 문학 문제 출제 전문가입니다. 주어진 작품을 분석하고 수능 형식에 맞는 양질의 문제를 생성해주세요."
        )
        
        explanation_text = "[해설]\n(정답 선택지에 대한 해설 및 나머지 선택지가 오답인 이유)" if add_explanation else ""

        prompt = f"""
        다음 문학 작품을 바탕으로 "{question_type}" 유형의 수능 국어 문학 문제를 생성해주세요.
        
        작품:{title_info}{author_info}
        {text}
        
        문제 생성 지침:
        1. 문제 유형: {question_type}
        2. 질문 형식: "{question_pattern}"
        3. 객관식 5지선다형으로 작성
        4. {example_instruction}작품의 내용을 정확히 이해해야 풀 수 있는 문제 생성
        5. 모든 선택지는 작품과 연관된 내용으로 작성
        6. 정답은 반드시 작품에서 근거를 찾을 수 있어야 함
        7. 오답은 그럴듯하게 작성하되, 확실히 틀린 내용이어야 함
        8. 특정 구절이나 소재를 지목하는 문제의 경우, 작품 속 해당 부분을 명확히 지시해야 함
        
        작품 분석 정보:
        - 장르: {analysis['genre']}
        - 주제: {analysis['theme']}
        - 핵심 내용: {analysis['summary']}
        - 주요 인물: {', '.join(analysis['characters']) if analysis['characters'] else 'N/A'}
        - 표현 기법: {', '.join(analysis['literary_devices'])}
        - 구성 및 서술: {analysis['narrative_structure']}
        - 주요 소재/이미지: {', '.join(analysis['symbols_and_images'])}
        
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
        
        {explanation_text}
        """
        
        human_message = HumanMessage(content=prompt)
        
        try:
            response = self.llm([system_message, human_message])
            return response.content
        
        except Exception as e:
            print(f"문제 생성 중 오류 발생: {e}")
            return f"문제 생성 중 오류가 발생했습니다: {str(e)}"
    
    def generate_comparative_question(self, 
                                    texts: List[str], 
                                    question_type: str = "비교와 대조", 
                                    analyses: List[Dict[str, Any]] = None,
                                    titles: List[str] = None,
                                    authors: List[str] = None,
                                    include_negation: bool = None, 
                                    specific_pattern: str = None,
                                    add_explanation: bool = True) -> str:
        """
        두 개 이상의 문학 작품을 비교하는 문제 생성
        
        Args:
            texts: 문학 작품 텍스트 리스트
            question_type: 문제 유형 (주로 "비교와 대조")
            analyses: 작품 분석 결과 리스트 (없으면 자동 분석)
            titles: 작품 제목 리스트 (없으면 None)
            authors: 작가 리스트 (없으면 None)
            include_negation: 부정형 문제 여부 (None이면 자동 결정)
            specific_pattern: 특정 문제 패턴 (None이면 자동 선택)
            add_explanation: 해설 추가 여부
            
        Returns:
            str: 생성된 문제
        """
        if len(texts) < 2:
            raise ValueError("비교 문제를 생성하려면 최소 2개 이상의 작품이 필요합니다.")
        
        # 분석 결과가 없으면 작품들 자동 분석
        if analyses is None:
            analyses = []
            for i, text in enumerate(texts):
                title = titles[i] if titles and i < len(titles) else None
                author = authors[i] if authors and i < len(authors) else None
                analyses.append(self.analyze_literature(text, title, author))
        
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
            question_pattern = "(가)와 (나)의 공통점으로 가장 적절한 것은?"
        
        # 문제 작성을 위한 작품 정보 준비
        work_texts = []
        work_analyses = []
        
        for i, (text, analysis) in enumerate(zip(texts, analyses)):
            work_label = f"({chr(97 + i)})"  # (a), (b), (c), ...
            title_info = f"제목: {titles[i]}" if titles and i < len(titles) else ""
            author_info = f"작가: {authors[i]}" if authors and i < len(authors) else ""
            
            work_text = f"{work_label}\n{title_info}\n{author_info}\n{text}"
            work_texts.append(work_text)
            
            work_analysis = f"작품 {work_label} 분석:\n"
            work_analysis += f"- 장르: {analysis['genre']}\n"
            work_analysis += f"- 주제: {analysis['theme']}\n"
            work_analysis += f"- 핵심 내용: {analysis['summary']}\n"
            work_analysis += f"- 주요 표현 기법: {', '.join(analysis['literary_devices'])}\n"
            work_analysis += f"- 구성 및 서술: {analysis['narrative_structure']}\n"
            
            work_analyses.append(work_analysis)
        
        combined_work_texts = "\n\n".join(work_texts)
        combined_work_analyses = "\n\n".join(work_analyses)
        
        system_message = SystemMessage(
            content="당신은 수능 국어 문학 문제 출제 전문가입니다. 주어진 여러 작품을 비교 분석하고 수능 형식에 맞는 양질의 문제를 생성해주세요."
        )
        
        explanation_text = "[해설]\n(정답 선택지에 대한 해설 및 나머지 선택지가 오답인 이유)" if add_explanation else ""

        prompt = f"""
        다음 문학 작품들을 바탕으로 "{question_type}" 유형의 수능 국어 문학 문제를 생성해주세요.
        
        {combined_work_texts}
        
        문제 생성 지침:
        1. 문제 유형: {question_type}
        2. 질문 형식: "{question_pattern}"
        3. 객관식 5지선다형으로 작성
        4. 작품들의 공통점과 차이점을 정확히 파악해야 풀 수 있는 문제 생성
        5. 모든 선택지는 작품들과 연관된 내용으로 작성
        6. 정답은 반드시 작품들에서 근거를 찾을 수 있어야 함
        7. 오답은 그럴듯하게 작성하되, 확실히 틀린 내용이어야 함
        8. 작품 간의 의미 있는 비교점을 중심으로 문제 출제
        
        작품 분석 정보:
        {combined_work_analyses}
        
        문제 작성 형식:
        [문제]
        {question_pattern}
        
        [선택지]
        ① (선택지 1)
        ② (선택지 2)
        ③ (선택지 3)
        ④ (선택지 4)
        ⑤ (선택지 5)
        
        [정답]
        (정답 번호)
        
        {explanation_text}
        """
        
        human_message = HumanMessage(content=prompt)
        
        try:
            response = self.llm([system_message, human_message])
            return response.content
        
        except Exception as e:
            print(f"비교 문제 생성 중 오류 발생: {e}")
            return f"비교 문제 생성 중 오류가 발생했습니다: {str(e)}"
    
    def generate_external_reference_question(self, 
                                           text: str, 
                                           reference_info: str,
                                           analysis: Dict[str, Any] = None,
                                           title: str = None,
                                           author: str = None,
                                           include_negation: bool = None, 
                                           specific_pattern: str = None,
                                           add_explanation: bool = True) -> str:
        """
        외적 준거를 활용한 문제 생성
        
        Args:
            text: 문학 작품 텍스트
            reference_info: 외적 준거 정보 (작가 정보, 문학사적 맥락, 비평 관점 등)
            analysis: 작품 분석 결과 (없으면 자동 분석)
            title: 작품 제목 (없으면 None)
            author: 작가 (없으면 None)
            include_negation: 부정형 문제 여부 (None이면 자동 결정)
            specific_pattern: 특정 문제 패턴 (None이면 자동 선택)
            add_explanation: 해설 추가 여부
            
        Returns:
            str: 생성된 문제
        """
        question_type = "외적 준거"
        
        # 분석 결과가 없으면 작품 자동 분석
        if analysis is None:
            analysis = self.analyze_literature(text, title, author)
        
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
            question_pattern = "<보기>를 바탕으로 윗글을 감상한 내용으로 적절하지 않은 것은?"
        
        title_info = f"\n제목: {title}" if title else ""
        author_info = f"\n작가: {author}" if author else ""
        
        system_message = SystemMessage(
            content="당신은 수능 국어 문학 문제 출제 전문가입니다. 주어진 작품과 외적 준거를 활용하여 수능 형식에 맞는 양질의 문제를 생성해주세요."
        )
        
        explanation_text = "[해설]\n(정답 선택지에 대한 해설 및 나머지 선택지가 오답인 이유)" if add_explanation else ""

        prompt = f"""
        다음 문학 작품과 <보기>를 바탕으로 "외적 준거" 유형의 수능 국어 문학 문제를 생성해주세요.
        
        작품:{title_info}{author_info}
        {text}
        
        <보기>
        {reference_info}
        
        문제 생성 지침:
        1. 문제 유형: 외적 준거
        2. 질문 형식: "{question_pattern}"
        3. 객관식 5지선다형으로 작성
        4. 작품과 <보기>를 연결하여 해석하는 문제 생성
        5. 모든 선택지는 작품과 <보기>의 연관성에 기반하여 작성
        6. 정답은 반드시 작품과 <보기>에서 근거를 찾을 수 있어야 함
        7. 오답은 그럴듯하게 작성하되, 확실히 틀린 내용이어야 함
        8. <보기>의 관점을 잘못 적용하거나 작품 내용을 잘못 해석한 것을 오답으로 설정
        
        작품 분석 정보:
        - 장르: {analysis['genre']}
        - 주제: {analysis['theme']}
        - 핵심 내용: {analysis['summary']}
        - 주요 인물: {', '.join(analysis['characters']) if analysis['characters'] else 'N/A'}
        - 표현 기법: {', '.join(analysis['literary_devices'])}
        - 구성 및 서술: {analysis['narrative_structure']}
        - 주요 소재/이미지: {', '.join(analysis['symbols_and_images'])}
        
        문제 작성 형식:
        [문제]
        {question_pattern}
        
        [보기]
        {reference_info}
        
        [선택지]
        ① (선택지 1)
        ② (선택지 2)
        ③ (선택지 3)
        ④ (선택지 4)
        ⑤ (선택지 5)
        
        [정답]
        (정답 번호)
        
        {explanation_text}
        """
        
        human_message = HumanMessage(content=prompt)
        
        try:
            response = self.llm([system_message, human_message])
            return response.content
        
        except Exception as e:
            print(f"외적 준거 문제 생성 중 오류 발생: {e}")
            return f"외적 준거 문제 생성 중 오류가 발생했습니다: {str(e)}"
    
    def generate_question_set(self, 
                            text: str, 
                            num_questions: int = 3, 
                            title: str = None, 
                            author: str = None) -> Dict[str, Any]:
        """
        문학 작품에 대한 여러 유형의 문제 세트 생성
        
        Args:
            text: 문학 작품 텍스트
            num_questions: 생성할 문제 수
            title: 작품 제목 (없으면 None)
            author: 작가 (없으면 None)
            
        Returns:
            Dict: 분석 결과와 생성된 문제들
        """
        # 작품 분석
        analysis = self.analyze_literature(text, title, author)
        
        results = {
            "analysis": analysis,
            "questions": []
        }
        
        # 분석 결과에서 추천된 문제 유형 사용
        question_types = analysis.get("recommended_question_types", ["작품 이해", "표현상 특징", "인물 분석", "작가 의도"])
        
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
                text=text,
                question_type=question_type,
                analysis=analysis,
                include_negation=include_negation,
                title=title,
                author=author
            )
            
            results["questions"].append({
                "question_type": question_type,
                "content": question
            })
        
        return results
    
    def generate_comparative_question_set(self, 
                                       texts: List[str], 
                                       num_questions: int = 3, 
                                       titles: List[str] = None, 
                                       authors: List[str] = None) -> Dict[str, Any]:
        """
        여러 문학 작품에 대한 비교 문제 세트 생성
        
        Args:
            texts: 문학 작품 텍스트 리스트
            num_questions: 생성할 문제 수
            titles: 작품 제목 리스트 (없으면 None)
            authors: 작가 리스트 (없으면 None)
            
        Returns:
            Dict: 분석 결과와 생성된 문제들
        """
        if len(texts) < 2:
            raise ValueError("비교 문제를 생성하려면 최소 2개 이상의 작품이 필요합니다.")
        
        # 작품 분석
        analyses = []
        for i, text in enumerate(texts):
            title = titles[i] if titles and i < len(titles) else None
            author = authors[i] if authors and i < len(authors) else None
            analyses.append(self.analyze_literature(text, title, author))
        
        results = {
            "analyses": analyses,
            "questions": []
        }
        
        # 비교 문제 유형 선택
        question_types = ["비교와 대조"] * max(1, num_questions // 3)
        question_types.extend(["작품 이해", "표현상 특징", "인물 분석", "작가 의도", "화자/서술자", 
                              "배경 분석", "상징과 이미지", "시어와 구절"])
        
        # 문제 수에 맞게 유형 조정
        question_types = question_types[:num_questions]
        
        # 문제 생성
        for i, question_type in enumerate(question_types):
            # 부정형과 긍정형 문제 번갈아 생성
            include_negation = i % 2 == 0
            
            if question_type == "비교와 대조":
                question = self.generate_comparative_question(
                    texts=texts,
                    question_type=question_type,
                    analyses=analyses,
                    include_negation=include_negation,
                    titles=titles,
                    authors=authors
                )
            else:
                # 작품 중 하나를 선택하여 개별 문제 생성
                work_idx = i % len(texts)
                question = self.generate_question(
                    text=texts[work_idx],
                    question_type=question_type,
                    analysis=analyses[work_idx],
                    include_negation=include_negation,
                    title=titles[work_idx] if titles and work_idx < len(titles) else None,
                    author=authors[work_idx] if authors and work_idx < len(authors) else None
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
            filename = f"ksat_literature_questions_{timestamp}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            # 단일 작품 분석인지 비교 분석인지 확인
            if "analysis" in results:
                # 단일 작품 분석 결과 저장
                analysis = results["analysis"]
                f.write("=== 작품 분석 결과 ===\n")
                f.write(f"장르: {analysis['genre']}\n")
                f.write(f"주제: {analysis['theme']}\n")
                f.write(f"핵심 내용: {analysis['summary']}\n")
                
                if analysis.get('characters'):
                    f.write(f"주요 인물: {', '.join(analysis['characters'])}\n")
                
                f.write(f"표현 기법: {', '.join(analysis['literary_devices'])}\n")
                f.write(f"구성 및 서술: {analysis['narrative_structure']}\n")
                f.write(f"주요 소재/이미지: {', '.join(analysis['symbols_and_images'])}\n")
                f.write(f"추천 문제 유형: {', '.join(analysis['recommended_question_types'])}\n")
                
                if analysis.get('rationale'):
                    f.write(f"추천 이유: {analysis['rationale']}\n")
                
                f.write("\n")
            
            elif "analyses" in results:
                # 여러 작품 분석 결과 저장
                for i, analysis in enumerate(results["analyses"]):
                    f.write(f"=== 작품 {chr(97 + i)} 분석 결과 ===\n")
                    f.write(f"장르: {analysis['genre']}\n")
                    f.write(f"주제: {analysis['theme']}\n")
                    f.write(f"핵심 내용: {analysis['summary']}\n")
                    
                    if analysis.get('characters'):
                        f.write(f"주요 인물: {', '.join(analysis['characters'])}\n")
                    
                    f.write(f"표현 기법: {', '.join(analysis['literary_devices'])}\n")
                    f.write(f"구성 및 서술: {analysis['narrative_structure']}\n")
                    f.write(f"주요 소재/이미지: {', '.join(analysis['symbols_and_images'])}\n")
                    f.write(f"추천 문제 유형: {', '.join(analysis['recommended_question_types'])}\n")
                    
                    if analysis.get('rationale'):
                        f.write(f"추천 이유: {analysis['rationale']}\n")
                    
                    f.write("\n")
            
            # 각 문제 저장
            for i, question_data in enumerate(results["questions"]):
                f.write(f"=== 문제 {i+1} ({question_data['question_type']}) ===\n")
                f.write(f"{question_data['content']}\n\n")
        
        return filename

# 유용한 유틸리티 함수들
def extract_answers_from_question_set(question_set: Dict[str, Any]) -> List[str]:
    """
    문제 세트에서 정답만 추출
    
    Args:
        question_set: 문제 세트 (generate_question_set의 결과)
        
    Returns:
        List[str]: 정답 번호 리스트
    """
    answers = []
    
    for question_data in question_set["questions"]:
        content = question_data["content"]
        answer_section = content.split("[정답]")
        
        if len(answer_section) > 1:
            answer_line = answer_section[1].strip().split("\n")[0].strip()
            answers.append(answer_line)
        else:
            answers.append("정답 추출 실패")
    
    return answers

def mark_literary_elements(text: str, elements_to_mark: List[str], markers: List[str] = None) -> str:
    """
    문학 작품에서 특정 요소들(소재, 표현, 시어 등)을 마킹
    
    Args:
        text: 원본 텍스트
        elements_to_mark: 마킹할 요소들의 리스트
        markers: 사용할 마커들 (기본값: ㉠, ㉡, ㉢, ...)
        
    Returns:
        str: 요소들이 마킹된 텍스트
    """
    if markers is None:
        markers = ['㉠', '㉡', '㉢', '㉣', '㉤', '㉥', '㉦', '㉧', '㉨', '㉩']
    
    marked_text = text
    
    for i, element in enumerate(elements_to_mark):
        if i < len(markers):
            # 요소의 첫 등장만 마킹 (단어 경계 고려)
            pattern = r'(?<!\w)' + re.escape(element) + r'(?!\w)'
            if re.search(pattern, marked_text):
                marked_text = re.sub(pattern, markers[i], marked_text, count=1)
    
    return marked_text

def mark_important_sentences(text: str, num_sentences: int = 3, marker: str = "밑줄") -> str:
    """
    문학 작품에서 중요 문장에 밑줄 표시
    
    Args:
        text: 원본 텍스트
        num_sentences: 표시할 문장 수
        marker: 사용할 마커 설명 (기본값: "밑줄")
        
    Returns:
        str: 중요 문장이 표시된 텍스트와 표시된 문장 리스트
    """
    # 문장 경계 인식 패턴
    sentence_pattern = r'[^.!?]+[.!?]+'
    sentences = re.findall(sentence_pattern, text)
    
    if len(sentences) <= num_sentences:
        # 문장 수가 적으면 모든 문장 선택
        selected_indices = list(range(len(sentences)))
    else:
        # 균등하게 분산되도록 문장 선택
        step = len(sentences) // num_sentences
        selected_indices = [i * step for i in range(num_sentences)]
        
        # 마지막 선택 인덱스가 너무 크면 조정
        if selected_indices[-1] >= len(sentences):
            selected_indices[-1] = len(sentences) - 1
    
    # 선택된 문장 마킹
    marked_text = text
    marked_sentences = []
    
    for i in sorted(selected_indices, reverse=True):  # 뒤에서부터 처리하여 인덱스 문제 방지
        if i < len(sentences):
            sentence = sentences[i]
            marked_sentence = f"[{marker}] {sentence} [{marker} 끝]"
            marked_text = marked_text.replace(sentence, marked_sentence, 1)
            marked_sentences.append(sentence.strip())
    
    return marked_text, marked_sentences

def generate_boilerplate_external_reference(analysis: Dict[str, Any], 
                                           reference_type: str = "작가_정보") -> str:
    """
    외적 준거 문제를 위한 기본 보기 생성
    
    Args:
        analysis: 작품 분석 결과
        reference_type: 보기 유형 (작가_정보, 문학사적_맥락, 비평_관점, 사회문화적_배경)
        
    Returns:
        str: 생성된 보기 텍스트
    """
    genre = analysis.get('genre', '현대문학')
    theme = analysis.get('theme', '삶과 인간')
    
    if reference_type == "작가_정보":
        return f"""이 작품의 작가는 {genre}의 대표적 작가로, 주로 '{theme}'과 관련된 주제를 다루었다. 
작가는 자신의 개인적 경험을 바탕으로 작품 세계를 구축했으며, 특유의 감성과 언어 감각으로 
독자들에게 깊은 인상을 남겼다. 작가의 다른 작품들도 이 작품과 유사한 주제의식과 표현 방식을 
보여주고 있어 작가의 일관된 문학적 세계관을 엿볼 수 있다."""

    elif reference_type == "문학사적_맥락":
        return f"""이 작품이 쓰여진 시기는 한국 {genre}의 발전기로, 전통적 형식에서 벗어나 
새로운 문학적 실험이 활발하게 이루어지던 때였다. 특히 '{theme}'에 대한 
탐구는 당시 문학계의 중요한 흐름이었으며, 이 작품은 그러한 흐름 속에서 
독창적인 위치를 차지하고 있다. 이 시기의 다른 작품들과 비교해볼 때, 
이 작품의 문학사적 가치와 의의를 더 명확히 파악할 수 있다."""

    elif reference_type == "비평_관점":
        return f"""문학 작품은 표면적 의미 외에도 다양한 층위의 의미를 내포하고 있다. 
특히 '{theme}'을 다루는 작품은 인간의 본질적 고민을 반영하는 경우가 많다.
독자는 작품 속 표현과 구조를 통해 작가가 전달하고자 하는 메시지를 해석하며,
이 과정에서 독자 자신의 경험과 가치관이 작용하게 된다. 따라서 문학 작품의
의미는 작가의 의도뿐만 아니라 독자의 해석에 의해서도 풍부해질 수 있다."""

    elif reference_type == "사회문화적_배경":
        return f"""이 작품이 창작된 시대적 배경은 사회적, 정치적 변화가 급격하게 일어나던 
시기였다. 특히 '{theme}'과 관련된 사회적 인식의 변화는 작품의 주제 형성에 
중요한 영향을 미쳤다. 당시의 사회문화적 맥락을 이해하면 작품에 담긴 작가의 
문제의식과 작품이 독자들에게 전달하고자 하는 메시지를 더 깊이 있게 파악할 수 있다."""

    else:
        return f"""문학 작품은 작가, 작품, 독자의 상호작용을 통해 의미가 생성된다. 
'{theme}'이라는 주제는 시대와 배경에 따라 다양한 맥락에서 해석될 수 있으며, 
이 작품 역시 다양한 관점에서 분석이 가능하다. 작품의 의미를 온전히 이해하기 위해서는 
작품 자체의 내적 구조뿐만 아니라 작품을 둘러싼 외적 맥락도 함께 고려해야 한다."""

def read_passage_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        passage = f.read()
    return passage