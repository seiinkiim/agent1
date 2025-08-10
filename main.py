# app.py
# 상단에 한 줄 추가
import re
import os
import time
import random  # 인증번호용
import streamlit as st
import utils
# LangChain core
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# RAG: data load / split / embed / vectorstore
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # 환경 호환용
from langchain_community.vectorstores import FAISS

# 너의 스트리밍 핸들러
from utils import StreamHandler


# ---------------------------
# Streamlit 기본 설정
# ---------------------------
st.set_page_config(page_title="운동화 쇼핑 에이전트")
st.title("운동화 쇼핑 에이전트")

# API KEY
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# 세션 상태 기본값
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "store" not in st.session_state:
    st.session_state["store"] = dict()


# ---------------------------
# RAG 파이프라인 준비
# ---------------------------
@st.cache_resource(show_spinner=True)
def build_retriever(csv_path: str):
    # 1) Load
    loader = CSVLoader(csv_path, encoding="utf-8")
    docs = loader.load()

    # 2) Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = splitter.split_documents(docs)

    # 3) Embeddings (가벼운 모델)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4) VectorStore (FAISS)
    vs = FAISS.from_documents(splits, embedding=embedding)
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    return retriever

retriever = build_retriever("shoes_top12.csv")


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state["store"]:
        st.session_state["store"][session_id] = ChatMessageHistory()
    return st.session_state["store"][session_id]


def build_query_from_history_and_input(history: BaseChatMessageHistory, user_input: str, max_turns: int = 4) -> str:
    """간단 쿼리 빌더: 최근 멀티턴 내용을 압축해 retriever용 쿼리 생성"""
    msgs = history.messages[-max_turns*2:] if hasattr(history, "messages") else []
    hist_text = []
    for m in msgs:
        role = getattr(m, "type", getattr(m, "role", ""))
        content = getattr(m, "content", "")
        if role in ("human", "user", "ai", "assistant"):
            hist_text.append(f"{role}: {content}")
    hist_blob = "\n".join(hist_text)
    query = f"{hist_blob}\nuser: {user_input}\n\n요약 키워드: 운동 목적, 쿠션, 통풍, 경량/안정, 굽 높이, 브랜드 선호"
    return query


# ---------------------------
# 링크 유틸 / 사용자 답변 요약 / 설명 재작성
# ---------------------------
def _md_link(url: str, label: str = "구매링크") -> str:
    """URL을 [라벨](URL) 형태의 마크다운 링크로 변환 (스킴 보정 포함)"""
    if not url:
        return ""
    u = str(url).strip()
    if not re.match(r"^https?://", u, re.IGNORECASE):
        u = "http://" + u
    return f"[{label}]({u})"

def summarize_user_answers(history: BaseChatMessageHistory, max_turns: int = 8) -> str:
    """최근 human/user 발화만 모아 한 줄 요약 (너무 일반적인 트리거 문구는 제외)"""
    if not hasattr(history, "messages"):
        return ""
    answers = []
    for m in history.messages[-max_turns*2:]:
        role = getattr(m, "type", getattr(m, "role", ""))
        if role in ("human", "user"):
            txt = (getattr(m, "content", "") or "").strip()
            if txt in ("운동화 추천해줘", "추천해줘", "운동화 보여줘"):
                continue
            answers.append(txt)
    return " / ".join(answers[-6:])  # 최신 6개까지

def rewrite_description_with_answers(brand: str, name: str, price: str, original_desc: str, user_answers: str) -> str:
    """제품 원본 설명 + 사용자 답변을 근거로 1~2문장 재작성 (컨텍스트 외 정보 금지)"""
    if not os.environ.get("OPENAI_API_KEY"):
        return ""
    llm_small = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.4, max_tokens=120)
    system_msg = (
        "당신은 러닝화 추천 설명을 작성하는 카피라이터입니다. "
        "다음 제품 설명과 사용자 답변만을 근거로, 제품 강점을 사용자 답변과 직접 연결해 "
        "한국어로 1~2문장을 작성하세요. 과장/추측 금지. 재작성 문장만 출력하세요."
    )
    user_msg = (
        f"[사용자 답변 요약]\n{user_answers}\n\n"
        f"[제품 정보]\n브랜드: {brand}\n제품명: {name}\n가격: {price}\n"
        f"[원본 설명]\n{original_desc}"
    )
    try:
        resp = llm_small.invoke(
            [{"role": "system", "content": system_msg},
             {"role": "user", "content": user_msg}]
        )
        text = (getattr(resp, "content", "") or "").strip()
        return re.sub(r"\s+", " ", text)
    except Exception:
        return ""


# ---------------------------
# 컨텍스트 생성 (원본설명 + 재작성 + 구매링크 마크다운)
# ---------------------------
def join_docs_with_rewrite(docs, user_answers: str):
    """브랜드/제품명/가격/설명/구매링크를 한 줄로. 설명은 '원본 + 재작성'."""
    rows = []
    for d in docs:
        t = d.page_content

        def grab(field):
            m = re.search(rf"{field}\s*[:=]\s*(.+)", t)
            return m.group(1).strip() if m else ""

        brand = grab("브랜드")
        name  = grab("제품명")
        price = grab("가격")
        desc  = grab("제품설명")
        url   = grab("구매링크")

        if not url:
            continue  # 링크 없는 항목 제외

        link_md = _md_link(url, "구매링크")
        rewritten = rewrite_description_with_answers(brand, name, price, desc, user_answers)
        final_desc = f"{desc} {rewritten}".strip() if rewritten else desc

        rows.append(
            f"브랜드:{brand} | 제품명:{name} | 가격:{price} | 설명:{final_desc} | {link_md}"
        )
    return "\n".join(rows)


# ---------------------------
# 프롬프트 (워크플로 유지 + RAG 컨텍스트 주입)
# ---------------------------
SYSTEM_PROMPT = """# 작업 설명: 운동화 쇼핑 에이전트

## 역할
당신은 친절하고 전문적인 운동화 쇼핑 에이전트입니다.  
당신의 주요 목표는 사용자와의 멀티턴 대화를 통해 사용자의 니즈를 정확하게 파악한 후,  
가장 적합한 실제 운동화 제품 3개를 추천하는 것입니다.  
대화는 단계적으로 진행되며, 각 단계에서 사용자 응답을 변수로 저장해 다음 단계에서 활용합니다.

---
### 참고 컨텍스트 (RAG)
아래는 CSV 기반 문서 검색으로 찾은 관련 정보입니다.  
추천 시 가능한 한 이 정보를 활용하세요.  
컨텍스트에 포함된 **구매링크를 그대로 사용**하여 출력하세요.  
링크가 없는 제품은 추천하지 마세요.
{context}
---

## 대화 흐름 (Workflow)
규칙
- 각 단계는 반드시 프롬프트에 작성된 내용들을 따라주세요.
- 반드시 그대로 사용하라는 말이 있으면, 절대 수정하거나 변형하지 마세요. 그대로 출력해야합니다.
- 글씨 크기나 폰트를 바꾸지 마세요. 가장 기본 형식으로 출력하세요.

### 1단계: 대화 시작 & 1번째 질문
- 트리거 문장 예시: 사용자가 "추천해줘", "운동화 보여줘", "운동화 추천해줘" 등의 말을 하면 대화를 시작합니다.

1. 먼저 아래 문장을 출력하세요.

알겠습니다. 우선 몇 가지 질문을 통해 당신에게 맞는 운동화를 추천해드리겠습니다.

2. 1번째 질문을 출력합니다. 
----
규칙
-아래의 질문은 1번째 질문입니다. 절대 문장을 변형하거나 수정하지 마세요. 글씨 크기나 글꼴도 변경하지 마세요.
----
신발의 무게감에 대해 선호하는 스타일을 말씀해 주세요. 가벼운 편을 선호하시나요 , 혹은 무게감이 있어도 안정감을 주는 편이 좋으신가요?
---

### 2단계: 2번째 질문 출력
----
규칙
-아래의 질문은 2번째 질문입니다. 절대 문장을 변형하거나 수정하지 마세요. 글씨 크기나 글꼴도 변경하지 마세요.
----
알겠습니다. 운동화를 신었을 때 발이 답답하거나 열이 찬다고 느낀 적이 있다면, 어떤 기능이나 소재(예: 메쉬 소재, 통풍감 등)가 있으면 더 편안할지 말씀해 주세요.

---
### 3단계: 3번째 추가 조건 질문 출력
----
규칙
-아래의 질문은 3번째 질문입니다. 절대 문장을 변형하거나 수정하지 마세요. 글씨 크기나 글꼴도 변경하지 마세요.
----
알겠습니다. 발 앞부분이나 뒤꿈치에 가해지는 충격 완화 기능(쿠션)에 대해 어떻게 생각하시나요? 선호하는 쿠션감을 설명해주세요.
---
### 4단계: 4번째 추가 조건 질문 출력
규칙
-아래의 질문은 4번째 질문입니다. 절대 문장을 변형하거나 수정하지 마세요. 글씨 크기나 글꼴도 변경하지 마세요.
----
알겠습니다. 운동화의 높이에 대해 선호하는 편이 있나요? 평평한 디자인과 약간 굽이 있는 디자인 중 어떤 스타일을 선호하는지 말씀해주세요.
---
### 5단계: 1,2,3,4단계 질문 답변 기반 맞춤형 운동화 추천 제공
- 추가적인 질문은 절대 하지 마세요.
- 질문들과 사용자의 답변을 기반으로 맞춤형 추천을 해야합니다.
- 사용자의 응답을 기반으로 가장 적합한 실제 운동화 3개를 추천하세요.
- 반드시 아래 문장으로 시작해야 합니다.

시작 문장
알겠습니다. 말씀해주신 내용들을 기반으로 운동화를 추천합니다.

추천 형식 규칙 (절대 위반 금지)
- 추천시, 구매링크도 함께 제공하여 추천해주세요.
- 반드시 다음과 같은 형식으로 리스트 3개를 출력하세요:
  - `1. [브랜드] [제품명] | [가격] | [설명] | [구매링크](URL)`
  - `2. ...`
  - `3. ...`
- 각 줄은 숫자 순번(1., 2., 3.)으로 시작해야 하며, 줄바꿈된 목록 형태여야 합니다.
- 각 운동화는 실제 브랜드명, 제품명, 가격과 함께 한 줄 설명을 포함해야 합니다.
- **설명에는 반드시 사용자가 언급한 기능 또는 조건이 포함되어야 하며**, 컨텍스트의 설명(원본+재작성)을 근거로 쓰세요.
- **링크 텍스트는 반드시 '구매링크'**를 사용하고, 마크다운 링크(`[구매링크](URL)`)로 출력하세요.
- 컨텍스트에 제공된 링크를 그대로 사용하세요.

형식 예시 (참고용)
1. 나이키 에어줌 페가수스 40 | 129,000원 | 통풍성 있는 갑피와 안정적인 쿠션으로 장거리 주행에서 발 피로를 줄여줍니다. | [구매링크](https://example.com)

---
### 6단계: 대화 종료 

- 운동화 추천이 끝났음을 사용자에게 명확하게 알립니다.
- 반드시 아래 문장을 그대로 출력합니다. (글자, 띄어쓰기, 문장 부호를 절대 변경하지 마세요.)
    운동화 추천이 종료되었습니다! 

"""

# ChatPromptTemplate 구성
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# 모델(스트리밍)
llm = ChatOpenAI(model_name="gpt-4o", streaming=True)

# Runnable + 메모리
chain = prompt | llm | StrOutputParser()
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


# ---------------------------
# 이전 대화 출력
# ---------------------------
if len(st.session_state["messages"]) > 0:
    for role, msg in st.session_state["messages"]:
        # assistant 메시지는 마크다운으로 렌더 → [구매링크](URL) 클릭 가능
        if role == "assistant":
            st.chat_message(role).markdown(msg)
        else:
            st.chat_message(role).write(msg)


# ---------------------------
# 입력 & 응답
# ---------------------------
if user_input := st.chat_input("메시지를 입력해 주세요"):
    # 사용자 메시지 표시/저장
    st.chat_message("user").write(user_input)
    st.session_state["messages"].append(("user", user_input))

    # RAG 컨텍스트 생성 (사용자 답변 요약 → 제품 설명 재작성 포함)
    history = get_session_history("abc123")
    query = build_query_from_history_and_input(history, user_input)
    rag_docs = retriever.get_relevant_documents(query)
    user_answers = summarize_user_answers(history)
    context = join_docs_with_rewrite(rag_docs, user_answers)

    # 응답(스트리밍)
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        response = chain_with_memory.invoke(
            {"question": user_input, "context": context},
            config={"configurable": {"session_id": "abc123"}, "callbacks": [stream_handler]},
        )

    # 최종 텍스트 저장
    st.session_state["messages"].append(("assistant", response))

    # 응답에 종료 문구가 있으면 즉시 인증번호 출력
    if "운동화 추천이 종료되었습니다!" in response:
        code = f"{random.randint(0, 9999):04d}"
        end_msg = f"인증번호: {code}"
        st.chat_message("assistant").write(end_msg)
        st.session_state["messages"].append(("assistant", end_msg))
