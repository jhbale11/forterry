import streamlit as st
import os
import json
import requests
from ocr_processor import process_ocr_response_for_textbook, chunk_text, correct_text_with_llm
from ksat_nonlit import KSATQuestionGenerator
from ksat_literature import KSATLiteratureQuestionGenerator
from langchain_openai import ChatOpenAI

# Streamlit 페이지 설정
st.set_page_config(page_title="KSAT 문제 생성기", layout="wide")

# 사이드바에서 API 키 입력
st.sidebar.title("API 키 설정")
upstage_api_key = st.sidebar.text_input("Upstage API 키", type="password")
openai_api_key = st.sidebar.text_input("OpenAI API 키", type="password")
anthropic_api_key = st.sidebar.text_input("Anthropic API 키", type="password")

# API 키 유효성 검사
def check_api_keys(page):
    if page == "OCR 처리" and not upstage_api_key:
        st.error("OCR 처리를 위해 Upstage API 키를 입력해주세요.")
        return False
    if not openai_api_key and not anthropic_api_key:
        st.error("문제 생성을 위해 OpenAI 또는 Anthropic API 키를 입력해주세요.")
        return False
    return True

# 사이드바에서 페이지 선택
st.sidebar.title("기능 선택")
page = st.sidebar.radio("페이지를 선택하세요:", ["OCR 처리", "비문학 문제 생성", "문학 문제 생성"])

# 출력 폴더 생성
os.makedirs("output", exist_ok=True)

# OCR 처리 페이지
if page == "OCR 처리":
    st.title("PDF OCR 처리")
    st.write("PDF 파일을 업로드하여 OCR 결과를 TXT 파일로 저장합니다.")

    if not check_api_keys(page):
        st.stop()

    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])
    if uploaded_file is not None:
        # 업로드된 파일 저장
        input_pdf = os.path.join("output", uploaded_file.name)
        with open(input_pdf, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 출력 폴더 및 파일 설정
        input_filename = os.path.basename(input_pdf)
        input_name_without_ext = os.path.splitext(input_filename)[0]
        output_folder = os.path.join("output", input_name_without_ext)
        os.makedirs(output_folder, exist_ok=True)
        output_txt = os.path.join(output_folder, f"(result){input_name_without_ext}.txt")

        # OCR 처리
        if st.button("OCR 처리 시작"):
            with st.spinner("OCR 처리 중..."):
                # Step 1: OCR API 호출
                url = "https://api.upstage.ai/v1/document-ai/ocr"
                headers = {"Authorization": f"Bearer {upstage_api_key}"}
                with open(input_pdf, "rb") as file:
                    files = {"document": file}
                    response = requests.post(url, headers=headers, files=files)
                if response.status_code != 200:
                    st.error("OCR API 호출 실패")
                    st.stop()
                ocr_result = response.json()
                ocr_result_path = os.path.join(output_folder, "ocr_result.json")
                with open(ocr_result_path, "w", encoding="utf-8") as f:
                    json.dump(ocr_result, f, ensure_ascii=False, indent=2)
                st.subheader("중간 단계 1: OCR 원본 결과")
                st.json(ocr_result)
                st.download_button(
                    label="OCR 원본 결과 다운로드",
                    data=json.dumps(ocr_result, ensure_ascii=False, indent=2),
                    file_name="ocr_result.json",
                    mime="application/json"
                )

                # Step 2: OCR 결과에서 텍스트 추출
                full_text = process_ocr_response_for_textbook(ocr_result)
                processed_text_path = os.path.join(output_folder, "processed_text.txt")
                with open(processed_text_path, "w", encoding="utf-8") as f:
                    f.write(full_text)
                st.subheader("중간 단계 2: 추출된 텍스트")
                st.text_area("추출된 텍스트", full_text, height=200)
                st.download_button(
                    label="추출된 텍스트 다운로드",
                    data=full_text,
                    file_name="processed_text.txt",
                    mime="text/plain"
                )

                # Step 3: LLM을 사용한 텍스트 보정
                text_chunks = chunk_text(full_text)
                chat_model = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0.3,
                    max_tokens=8000,
                    api_key=openai_api_key
                )
                corrected_chunks = []
                for i, chunk in enumerate(text_chunks):
                    st.write(f"청크 {i+1}/{len(text_chunks)} 보정 중...")
                    corrected_chunk = correct_text_with_llm(chunk, chat_model)
                    corrected_chunks.append(corrected_chunk)
                corrected_text = "\n\n".join(corrected_chunks)
                with open(output_txt, "w", encoding="utf-8") as f:
                    f.write(corrected_text)
                st.subheader("중간 단계 3: 보정된 텍스트")
                st.text_area("보정된 텍스트", corrected_text, height=300)
                st.download_button(
                    label="보정된 텍스트 다운로드",
                    data=corrected_text,
                    file_name=f"(result){input_name_without_ext}.txt",
                    mime="text/plain"
                )

# 비문학 문제 생성 페이지
elif page == "비문학 문제 생성":
    st.title("비문학 문제 생성")
    st.write("지문을 입력하거나 파일을 업로드하여 비문학 문제를 생성합니다.")

    if not check_api_keys(page):
        st.stop()

    # 지문 입력 방법 선택
    input_method = st.radio("지문 입력 방법:", ["직접 입력", "TXT 파일 업로드"])

    passage = None
    if input_method == "직접 입력":
        passage = st.text_area("지문을 입력하세요:", height=300)
    else:
        uploaded_file = st.file_uploader("TXT 파일을 업로드하세요", type=["txt"])
        if uploaded_file is not None:
            passage = uploaded_file.read().decode("utf-8")
            st.text_area("업로드된 지문", passage, height=300)

    num_questions = st.number_input("생성할 문제 수", min_value=1, max_value=10, value=3)

    # 제공자 선택 (OpenAI 또는 Anthropic)
    provider = st.selectbox("LLM 제공자 선택", ["openai", "anthropic"])
    selected_api_key = openai_api_key if provider == "openai" else anthropic_api_key

    if passage and st.button("문제 생성"):
        with st.spinner("문제 생성 중..."):
            generator = KSATQuestionGenerator(
                openai_api_key=openai_api_key,
                anthropic_api_key=anthropic_api_key,
                provider=provider
            )
            # Step 1: 지문 분석
            analysis = generator.analyze_passage(passage)
            st.subheader("중간 단계 1: 지문 분석 결과")
            st.json(analysis)
            analysis_file = os.path.join("output", "nonlit_analysis.json")
            with open(analysis_file, "w", encoding="utf-8") as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
            st.download_button(
                label="지문 분석 결과 다운로드",
                data=json.dumps(analysis, ensure_ascii=False, indent=2),
                file_name="nonlit_analysis.json",
                mime="application/json"
            )

            # Step 2: 문제 생성
            question_set = generator.generate_question_set(passage, num_questions)
            st.subheader("중간 단계 2: 생성된 문제")
            st.json(question_set)
            output_file = generator.save_to_file(question_set)
            with open(output_file, "r", encoding="utf-8") as f:
                result_text = f.read()
            st.text_area("생성된 문제", result_text, height=500)
            st.download_button(
                label="문제 다운로드",
                data=result_text,
                file_name=os.path.basename(output_file),
                mime="text/plain"
            )

# 문학 문제 생성 페이지
elif page == "문학 문제 생성":
    st.title("문학 문제 생성")
    st.write("지문을 입력하거나 파일을 업로드하여 문학 문제를 생성합니다.")

    if not check_api_keys(page):
        st.stop()

    # 지문 입력 방법 선택
    input_method = st.radio("지문 입력 방법:", ["직접 입력", "TXT 파일 업로드"])

    passage = None
    if input_method == "직접 입력":
        passage = st.text_area("지문을 입력하세요:", height=300)
    else:
        uploaded_file = st.file_uploader("TXT 파일을 업로드하세요", type=["txt"])
        if uploaded_file is not None:
            passage = uploaded_file.read().decode("utf-8")
            st.text_area("업로드된 지문", passage, height=300)

    num_questions = st.number_input("생성할 문제 수", min_value=1, max_value=10, value=3)
    question_types = st.multiselect(
        "문제 유형 선택 (선택하지 않으면 추천 유형 사용)",
        options=[
            "작품 이해", "표현상 특징", "서술상 특징", "작가 의도", "인물 분석",
            "갈등 양상", "상징과 이미지", "시어와 구절", "배경 분석", "비교와 대조",
            "작품 평가", "외적 준거", "화자/서술자"
        ],
        default=[]
    )

    title = st.text_input("작품 제목 (선택사항)")
    author = st.text_input("작가 (선택사항)")

    # 제공자 선택 (OpenAI 또는 Anthropic)
    provider = st.selectbox("LLM 제공자 선택", ["openai", "anthropic"])
    selected_api_key = openai_api_key if provider == "openai" else anthropic_api_key

    if passage and st.button("문제 생성"):
        with st.spinner("문제 생성 중..."):
            generator = KSATLiteratureQuestionGenerator(
                openai_api_key=openai_api_key,
                anthropic_api_key=anthropic_api_key,
                provider=provider
            )
            # Step 1: 문학 지문 분석
            analysis = generator.analyze_literature(passage, title, author)
            st.subheader("중간 단계 1: 문학 지문 분석 결과")
            st.json(analysis)
            analysis_file = os.path.join("output", "lit_analysis.json")
            with open(analysis_file, "w", encoding="utf-8") as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
            st.download_button(
                label="문학 지문 분석 결과 다운로드",
                data=json.dumps(analysis, ensure_ascii=False, indent=2),
                file_name="lit_analysis.json",
                mime="application/json"
            )

            # Step 2: 문제 생성
            if question_types:
                question_set = {"analysis": analysis, "questions": []}
                for i, q_type in enumerate(question_types[:num_questions]):
                    question = generator.generate_question(
                        text=passage,
                        question_type=q_type,
                        analysis=analysis,
                        include_negation=i % 2 == 0,
                        title=title,
                        author=author
                    )
                    question_set["questions"].append({"question_type": q_type, "content": question})
            else:
                question_set = generator.generate_question_set(passage, num_questions, title, author)

            st.subheader("중간 단계 2: 생성된 문제")
            st.json(question_set)
            output_file = generator.save_to_file(question_set)
            with open(output_file, "r", encoding="utf-8") as f:
                result_text = f.read()
            st.text_area("생성된 문제", result_text, height=500)
            st.download_button(
                label="문제 다운로드",
                data=result_text,
                file_name=os.path.basename(output_file),
                mime="text/plain"
            )