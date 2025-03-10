import requests
import os
import json
import re
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from collections import defaultdict

def detect_content_sections(words, page_width, page_height):
    """
    Detect and categorize content sections (passages, questions, answer choices) in a Korean textbook format.
    Uses spatial layout analysis to identify content sections and their relationships.
    """
    if not words:
        return []
    
    # Extract word data and add coordinate information
    word_data = []
    
    for word in words:
        box = word["boundingBox"]["vertices"]
        x_coords = [vertex.get("x", 0) for vertex in box]
        y_coords = [vertex.get("y", 0) for vertex in box]
        
        # Calculate center and dimensions
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        word_data.append({
            "text": word["text"],
            "center_x": center_x,
            "center_y": center_y,
            "min_x": min(x_coords),
            "max_x": max(x_coords),
            "min_y": min(y_coords),
            "max_y": max(y_coords),
            "width": width,
            "height": height,
            "original": word
        })
    
    # Determine column boundaries (assuming 2-column layout)
    mid_point = page_width / 2
    left_column_words = [w for w in word_data if w["center_x"] < mid_point]
    right_column_words = [w for w in word_data if w["center_x"] >= mid_point]
    
    # Function to group words into lines and then identify sections
    def process_column_words(column_words):
        if not column_words:
            return []
            
        # Sort words by vertical position
        column_words.sort(key=lambda w: w["center_y"])
        
        # Group words into lines based on Y-coordinate proximity
        avg_height = sum(w["height"] for w in column_words) / len(column_words)
        line_threshold = avg_height * 0.7  # 70% of average height as threshold
        
        lines = []
        current_line = [column_words[0]]
        
        for i in range(1, len(column_words)):
            prev_word = column_words[i-1]
            curr_word = column_words[i]
            
            # If Y difference is small enough, consider same line
            if abs(curr_word["center_y"] - prev_word["center_y"]) < line_threshold:
                current_line.append(curr_word)
            else:
                # Sort words in line by X-coordinate for proper reading order
                current_line.sort(key=lambda w: w["min_x"])  # Changed from center_x to min_x for better sorting
                lines.append(current_line)
                current_line = [curr_word]
        
        # Add the last line
        if current_line:
            current_line.sort(key=lambda w: w["min_x"])  # Changed from center_x to min_x
            lines.append(current_line)
        
        # Now identify sections from these properly ordered lines
        sections = []
        current_section = {"words": [], "type": "unknown", "start_y": 0, "end_y": 0}
        
        # Regex patterns for passage, question, and answer choice identifiers in Korean textbooks
        passage_patterns = [
            r'\[\d+-\d+\]',  # [06-12] format
            r'\[\d+\]',       # [36-40] format
            r'^\d+\.$',       # "12." format
            r'^\[지문\]'       # [지문] format
        ]
        
        question_patterns = [
            r'^\d+$',         # Just a number
            r'^문제\s*\d+',    # "문제 1" format
            r'^\d+\.\s',      # "1. " format
            r'^\d+\s',        # "06 " format (number with space)
            r'^\[문제\]'       # [문제] format
        ]
        
        answer_choice_patterns = [
            r'^①|^②|^③|^④|^⑤',  # Korean numbering
            r'^①\s|^②\s|^③\s|^④\s|^⑤\s',  # Korean numbering with space
            r'^보기',           # "보기" (answer choices)
            r'^(\(보기\))',     # "(보기)" format
            r'^\[보기\]'        # [보기] format
        ]
        
        # Process lines to identify sections
        for line_idx, line in enumerate(lines):
            # Check the first word in the line for section headers
            if not line:  # Skip empty lines
                continue
                
            # Combine all words in the line to check for headers in full text
            line_text = " ".join([w["text"] for w in line])
            first_word = line[0]
            text = first_word["text"].strip()
            
            # Check if this word or line might be a section header
            is_passage_header = any(re.match(pattern, text) for pattern in passage_patterns) or any(re.search(pattern, line_text) for pattern in passage_patterns)
            is_question_header = any(re.match(pattern, text) for pattern in question_patterns) or any(re.search(pattern, line_text) for pattern in question_patterns)
            is_answer_choice_header = any(re.match(pattern, text) for pattern in answer_choice_patterns) or any(re.search(pattern, line_text) for pattern in answer_choice_patterns)
            
            # Catch question numbers that might be missed by regex
            if not (is_passage_header or is_question_header or is_answer_choice_header):
                # Try to detect question numbers (just digits followed by a period)
                if text.isdigit() and len(text) <= 2:
                    # This is likely a question number
                    is_question_header = True
                # Check for parenthesized numbers like (1), (2)
                elif re.match(r'^\(\d+\)$', text):
                    is_question_header = True
            
            # Start a new section if header detected
            if is_passage_header or is_question_header or is_answer_choice_header:
                # Save current section if not empty
                if current_section["words"]:
                    if line_idx > 0:
                        current_section["end_y"] = max(w["max_y"] for w in lines[line_idx-1])
                    sections.append(current_section)
                
                # Start a new section
                section_type = "passage" if is_passage_header else ("question" if is_question_header else "answer_choice")
                current_section = {
                    "words": [],
                    "type": section_type,
                    "start_y": min(w["min_y"] for w in line),
                    "end_y": 0,
                    "header": text,
                    "line_text": line_text  # Store the full line text for context
                }
            
            # Add all words from this line to the current section
            current_section["words"].extend(line)
        
        # Add the last section
        if current_section["words"]:
            current_section["end_y"] = max(w["max_y"] for w in lines[-1]) if lines else 0
            sections.append(current_section)
        
        return sections
    
    # Process both columns separately using the improved line processing
    left_sections = process_column_words(left_column_words)
    right_sections = process_column_words(right_column_words)
    
    # Process all sections together but maintain column separation
    all_sections = []
    
    # Add column information to each section
    for section in left_sections:
        section["column"] = "left"
        all_sections.append(section)
    
    for section in right_sections:
        section["column"] = "right"
        all_sections.append(section)
    
    # Extract text with proper word ordering for each section
    organized_content = []
    
    for section in all_sections:
        # Improved line organization within each section
        section_words = section["words"]
        
        # Group section words into lines by Y coordinate
        avg_height = sum(w["height"] for w in section_words) / len(section_words)
        line_threshold = avg_height * 0.7
        
        # Sort words by Y coordinate first
        section_words.sort(key=lambda w: w["center_y"])
        
        # Group into lines
        lines = []
        current_line = [section_words[0]]
        
        for i in range(1, len(section_words)):
            prev_word = section_words[i-1]
            curr_word = section_words[i]
            
            if abs(curr_word["center_y"] - prev_word["center_y"]) < line_threshold:
                current_line.append(curr_word)
            else:
                lines.append(current_line)
                current_line = [curr_word]
        
        if current_line:
            lines.append(current_line)
        
        # Sort each line by X coordinate and join with spaces
        section_text_lines = []
        for line in lines:
            # Sort by X coordinate (left to right)
            line.sort(key=lambda w: w["min_x"])
            line_text = " ".join(word["text"] for word in line)
            section_text_lines.append(line_text)
        
        # Join lines with spaces to preserve paragraph structure
        section_text = " ".join(section_text_lines)
        
        organized_content.append({
            "type": section["type"],
            "text": section_text,
            "column": section["column"],
            "header": section.get("header", ""),
            "y_pos": section["start_y"]
        })
    
    return organized_content

def process_column_sections(column_sections):
    """
    Helper function to process sections within a column.
    Organizes passages, questions, and answer choices.
    """
    page_content = []
    
    # Organize sections by type for better grouping
    passages = []
    questions = []
    answer_choices = {}
    
    for section in column_sections:
        if section["type"] == "passage":
            passages.append(section)
        elif section["type"] == "question":
            # Extract question number if possible
            question_text = section["text"]
            question_number = None
            
            # Try to extract question number
            num_match = re.search(r'^\[?문제\]?\s*(\d+)|^(\d+)[.\s]', question_text)
            if num_match:
                question_number = num_match.group(1) or num_match.group(2)
            
            section["question_number"] = question_number
            questions.append(section)
        elif section["type"] == "answer_choice":
            # Try to associate with nearest question above
            nearest_question = None
            min_distance = float('inf')
            
            for q in questions:
                distance = section["y_pos"] - q["y_pos"]
                if 0 < distance < min_distance:  # Only consider questions above this answer choice
                    min_distance = distance
                    nearest_question = q
            
            if nearest_question and nearest_question.get("question_number"):
                q_num = nearest_question["question_number"]
                if q_num not in answer_choices:
                    answer_choices[q_num] = []
                answer_choices[q_num].append(section)
            else:
                # If can't determine question, use special key
                if "unknown" not in answer_choices:
                    answer_choices["unknown"] = []
                answer_choices["unknown"].append(section)
    
    # Combine passages into a single text
    if passages:
        combined_passage = "\n".join(["[지문] " + p["text"] for p in passages])
        page_content.append(combined_passage)
    
    # Process questions and their answers
    for question in questions:
        q_num = question.get("question_number", "00")
        
        # Format question text - clean up duplicated numbers
        question_text = question["text"]
        question_text = re.sub(r'^(\d+)\s+\1\s+', r'\1 ', question_text)
        
        # Replace "00" with the actual question number if available
        if q_num and "00" in question_text:
            question_text = question_text.replace("00", q_num, 1)
        
        formatted_question = f"[문제] {question_text}"
        page_content.append(formatted_question)
        
        # Add related answer choices
        if q_num in answer_choices:
            sorted_choices = sorted(answer_choices[q_num], key=lambda c: c["y_pos"])
            for choice in sorted_choices:
                formatted_choice = f"[보기] {choice['text']}"
                page_content.append(formatted_choice)
    
    # Add any "unknown" answer choices at the end
    if "unknown" in answer_choices:
        sorted_unknown = sorted(answer_choices["unknown"], key=lambda c: c["y_pos"])
        for choice in sorted_unknown:
            formatted_choice = f"[보기] {choice['text']}"
            page_content.append(formatted_choice)
    
    return page_content

def process_ocr_response_for_textbook(ocr_result):
    """
    Process OCR API response to extract and organize text, handling the specific
    layout of Korean language textbooks with passages, questions, and answer choices.
    """
    processed_content = []
    
    # Default page dimensions increased to ensure all content is captured
    default_width = 3000
    default_height = 4000
    
    # Process each page
    for page in ocr_result.get("pages", []):
        page_id = page.get("id", 0)
        page_width = page.get("width", default_width)
        page_height = page.get("height", default_height)
        
        # Skip if no word-level data
        if "words" not in page:
            processed_content.append({
                "page": page_id,
                "text": page.get("text", ""),
                "type": "unknown"
            })
            continue
        
        # Detect and organize content sections
        organized_sections = detect_content_sections(page["words"], page_width, page_height)
        
        # 명확하게 왼쪽 칼럼과 오른쪽 칼럼을 분리
        left_column_sections = [s for s in organized_sections if s["column"] == "left"]
        right_column_sections = [s for s in organized_sections if s["column"] == "right"]
        
        # 각 칼럼 내에서 수직 위치로 정렬
        left_column_sections.sort(key=lambda s: s["y_pos"])
        right_column_sections.sort(key=lambda s: s["y_pos"])
        
        # 왼쪽 칼럼 처리
        left_page_content = process_column_sections(left_column_sections)
        
        # 오른쪽 칼럼 처리
        right_page_content = process_column_sections(right_column_sections)
        
        # 왼쪽 칼럼 내용 다음에 오른쪽 칼럼 내용이 오도록 순서 지정
        page_content = left_page_content + right_page_content
        
        # Add to processed content
        processed_content.append({
            "page": page_id,
            "sections": page_content
        })
    
    # Join all content
    full_text = ""
    for page_content in processed_content:
        if "sections" in page_content:
            page_text = "\n\n".join(page_content["sections"])
        else:
            page_text = page_content.get("text", "")
        
        full_text += page_text + "\n\n--PAGE BREAK--\n\n"
    
    return full_text

def correct_text_with_llm(text_chunk, chat_model):
    """Correct a chunk of text using LLM."""
    prompt = f"""
아래 내용은 한국 국어 교재에서 OCR로 추출한 텍스트입니다. 이 텍스트를 자연스럽고 정확한 한국어로 교정해주세요.
교정 시 다음 사항을 따라주세요:

1. OCR 오류를 수정하고 깨진 문자나 글자를 복원하세요.

2. 다음 두 가지 형태의 내용을 모두 적절히 처리하세요:
   A. 지문-문제-보기 형태:
      - [지문]: 문제의 기반이 되는 읽기 자료입니다. 전체 지문을 하나로 통합하세요.
      - [문제]: 학생이 풀어야 할 질문입니다. 각 문제는 명확한 번호를 가져야 합니다.
      - [보기]: 답변 선택지는 반드시 한국어 번호(①, ②, ③, ④, ⑤)를 사용하여 정리하세요.
      
   B. 문제-답-해설 형태:
      - [문제]: 번호와 문제 내용을 포함합니다 (예: "0154 윗글에 대한 설명으로...").
      - [정답]: 정답을 명확히 표시합니다 (예: "정답: ④").
      - [해설]: 정답에 대한 해설과 오답에 대한 설명을 포함합니다.

3. 텍스트 구조를 일관되게 유지하세요:
   - 문제와 보기가 중복되는 경우 하나로 통합하세요.
   - 문제 번호와 문제 내용이 분리된 경우 적절히 통합하세요 (예: "06" + "윗글에 대한 설명으로..." → "06 윗글에 대한 설명으로...").
   - "00" 등으로 인식된 문제 번호가 있다면 문맥을 보고 적절한 번호로 수정하세요.

4. 문제-답-해설 형태일 경우 다음 형식으로 정리하세요:
   - [문제] 문제번호 문제내용
   - [정답] ④ (해당하는 번호)
   - [해설] 해설내용
   - [오답풀이] ① 오답1에 대한 설명 ② 오답2에 대한 설명... (있는 경우)

5. 모든 문제는 번호 순서대로 정렬하고, 각 문제 간에는 빈 줄을 추가하여 구분하세요.

6. 문장이 불완전하거나 의미가 모호한 경우 문맥을 고려하여 자연스럽게 복원하세요.

텍스트:
{text_chunk}
"""
    message = HumanMessage(content=prompt)
    response = chat_model([message])
    return response.content.strip()

def chunk_text(text, chunk_size=2000, overlap=200):
    """Split text into chunks with overlap to maintain context."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        # If we're not at the end of the text, find a good break point
        if end < text_length:
            # Try to break at a section boundary
            section_break = text.rfind('\n\n[', start, end)
            if section_break != -1 and section_break > start + chunk_size // 2:
                end = section_break
            else:
                # Try to break at a paragraph
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
                    end = paragraph_break + 2
                else:
                    # Try to break at a sentence
                    sentence_break = text.rfind('. ', start, end)
                    if sentence_break != -1 and sentence_break > start + chunk_size // 2:
                        end = sentence_break + 2
                    else:
                        # Just break at a space
                        space_break = text.rfind(' ', start, end)
                        if space_break != -1 and space_break > start + chunk_size // 2:
                            end = space_break + 1
        
        chunks.append(text[start:end])
        start = end - overlap if end < text_length else text_length
    
    return chunks

def process_pdf_with_ocr_and_correction(pdf_path, api_key, openai_api_key, output_path, output_folder):
    """Process PDF with OCR and correct with LLM."""
    url = "https://api.upstage.ai/v1/document-ai/ocr"
    headers = {"Authorization": f"Bearer {api_key}"}

    print(f"OCR 처리 중: {pdf_path}")
    with open(pdf_path, "rb") as file:
        files = {"document": file}
        response = requests.post(url, headers=headers, files=files)

    if response.status_code != 200:
        print(f"OCR API 오류: {response.status_code}")
        print(response.text)
        return False

    # OCR 결과를 JSON으로 파싱
    ocr_result = response.json()
    
    # OCR 결과 저장 (디버깅용)
    ocr_result_path = os.path.join(output_folder, "ocr_result.json")
    with open(ocr_result_path, "w", encoding="utf-8") as f:
        json.dump(ocr_result, f, ensure_ascii=False, indent=2)
    
    # 교재 형식에 맞게 텍스트 처리
    print("OCR 결과를 교재 형식에 맞게 처리 중...")
    full_text = process_ocr_response_for_textbook(ocr_result)
    
    # 처리된 원본 텍스트 저장 (디버깅용)
    print("처리된 원본 텍스트 저장 중...")
    processed_text_path = os.path.join(output_folder, "processed_text.txt")
    with open(processed_text_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    # 텍스트를 청크로 분할
    print("텍스트를 청크로 분할 중...")
    text_chunks = chunk_text(full_text)
    
    # ChatOpenAI 모델 초기화
    chat_model = ChatOpenAI(
        model="gpt-4o",  # model_name -> model
        temperature=0.3,
        max_tokens=8000,
        api_key=openai_api_key
    )
    
    # 각 청크를 LLM으로 보정
    print(f"총 {len(text_chunks)}개의 청크를 처리합니다.")
    corrected_chunks = []
    
    for i, chunk in enumerate(text_chunks):
        print(f"청크 {i+1}/{len(text_chunks)} 처리 중...")
        corrected_chunk = correct_text_with_llm(chunk, chat_model)
        corrected_chunks.append(corrected_chunk)
    
    # 보정된 청크를 하나로 합치기
    corrected_text = "\n\n".join(corrected_chunks)
    
    # 결과 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(corrected_text)
    
    print(f"텍스트가 {output_path} 파일로 저장되었습니다.")
    return True

def process_direct_ocr_json(json_data, openai_api_key, output_path, output_folder):
    """Process OCR JSON data directly without calling the OCR API."""
    # OCR 결과를 JSON으로 파싱
    ocr_result = json_data if isinstance(json_data, dict) else json.loads(json_data)
    
    # 교재 형식에 맞게 텍스트 처리
    print("OCR 결과를 교재 형식에 맞게 처리 중...")
    full_text = process_ocr_response_for_textbook(ocr_result)
    
    # 처리된 원본 텍스트 저장 (디버깅용)
    processed_text_path = os.path.join(output_folder, "processed_text.txt")
    with open(processed_text_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    # 텍스트를 청크로 분할
    print("텍스트를 청크로 분할 중...")
    text_chunks = chunk_text(full_text)
    
    # ChatOpenAI 모델 초기화
    chat_model = ChatOpenAI(
        model="gpt-4o",  # model_name -> model
        temperature=0.3,
        max_tokens=8000,
        api_key=openai_api_key
    )
    
    # 각 청크를 LLM으로 보정
    print(f"총 {len(text_chunks)}개의 청크를 처리합니다.")
    corrected_chunks = []
    
    for i, chunk in enumerate(text_chunks):
        print(f"청크 {i+1}/{len(text_chunks)} 처리 중...")
        corrected_chunk = correct_text_with_llm(chunk, chat_model)
        corrected_chunks.append(corrected_chunk)
    
    # 보정된 청크를 하나로 합치기
    corrected_text = "\n\n".join(corrected_chunks)
    
    # 결과 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(corrected_text)
    
    print(f"텍스트가 {output_path} 파일로 저장되었습니다.")
    return True