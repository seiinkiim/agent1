import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils import StreamHandler
import os, pandas as pd, hashlib, random, time

st.set_page_config(page_title="운동화 쇼핑 에이전트")
st.title("운동화 쇼핑 에이전트")

# API KEY
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "store" not in st.session_state:
    st.session_state["store"] = dict()
if "qa" not in st.session_state:
    st.session_state["qa"] = {"q1": None, "q2": None, "q3": None, "q4": None}
if "last_question_tag" not in st.session_state:
    st.session_state["last_question_tag"] = None  # "q1"|"q2"|"q3"|"q4"|None

# ====== 카탈로그 로드 & 버전/해시 ======
CATALOG_PATH = "starter_catalog_v1.csv"  # <-- 네 CSV 파일명
CATALOG_VER = "v1"
RULE_VER = "rule_v1.0"

@st.cache_data
def load_catalog(path):
    df = pd.read_csv(path)
    # 필수 컬럼 확인
    required_cols = {"id","brand","model","url","price_krw","stability","breathability","cushioning","category","notes"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV에 다음 컬럼이 필요합니다: {missing}")
    # 해시(재현성 기록용)
    cat_hash = hashlib.sha256(pd.util.hash_pandas_object(df.fillna(""), index=False).values).hexdigest()
    return df, cat_hash

try:
    catalog, catalog_hash = load_catalog(CATALOG_PATH)
except Exception as e:
    st.error(f"카탈로그 로드 오류: {e}")
    st.stop()

# ====== 질문 문구 고정(어시스턴트 출력 감지용) ======
Q1_TEXT = "신발은 가벼운 게 더 좋으세요, 아니면 약간 무게감이 있어도 안정적인 게 좋으세요?"
Q2_TEXT = "운동화 착용 중 발에 땀이 찼을 때 빠르게 건조되고 시원해지는 신발 기능이 필요하신가요?"
Q3_TEXT = "운동화 착용 중 발 앞이나 뒤꿈치에 가해지는 충격을 줄여주는 쿠션 기능이 필요하신가요?"
Q4_TEXT = "평평한 신발과 약간 높이가 있는 신발 중 어떤 높이의 운동화가 좋으신가요?"
START_SENTENCE = "알겠습니다. 말씀해주신 기능들 기반으로 운동화를 추천합니다."

# ====== 대화 기록 출력 ======
if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
    for role, message in st.session_state["messages"]:
        st.chat_message(role).write(message)

# ====== 세션 히스토리 핸들러 ======
def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    if session_ids not in st.session_state["store"]:
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]

# ====== 사용자 선호 파싱(룰베이스; 간단 키워드) ======
def parse_preferences_from_answers(qa_dict):
    """
    qa_dict: {"q1": str|None, "q2": str|None, "q3": str|None, "q4": str|None}
    반환: {"weight_pref","vent_pref","cushion_pref","height_pref"}
    값 도메인:
      weight_pref: light|balanced|stable
      vent_pref: low|medium|high
      cushion_pref: low|medium|high
      height_pref: flat|mid
    """
    # Q1 무게
    a1 = (qa_dict["q1"] or "").lower()
    if any(k in a1 for k in ["가벼", "light", "경량"]):
        weight_pref = "light"
    elif any(k in a1 for k in ["무게감", "안정", "stable", "묵직"]):
        weight_pref = "stable"
    else:
        weight_pref = "balanced"

    # Q2 통풍/건조
    a2 = (qa_dict["q2"] or "").lower()
    if any(k in a2 for k in ["통풍", "메쉬", "시원", "breath", "vent", "열 배출", "건조", "빠르게 마르"]):
        # 강도 추정
        if any(k in a2 for k in ["매우", "아주", "최대한", "필수", "중요", "high", "많이"]):
            vent_pref = "high"
        elif any(k in a2 for k in ["괜찮", "있으면 좋", "중간", "보통", "medium"]):
            vent_pref = "medium"
        else:
            vent_pref = "high"  # 언급 있으면 high로 가정
    else:
        vent_pref = "medium"

    # Q3 쿠션
    a3 = (qa_dict["q3"] or "").lower()
    if any(k in a3 for k in ["푹신", "부드", "쿠션", "충격 완화", "mellow", "soft", "max"]):
        if any(k in a3 for k in ["최대", "아주", "많", "high"]):
            cushion_pref = "high"
        elif any(k in a3 for k in ["적당", "중간", "보통", "medium"]):
            cushion_pref = "medium"
        else:
            cushion_pref = "high"
    elif any(k in a3 for k in ["단단", "반응", "탄성", "firm", "반발"]):
        cushion_pref = "low"
    else:
        cushion_pref = "medium"

    # Q4 높이
    a4 = (qa_dict["q4"] or "").lower()
    if any(k in a4 for k in ["평평", "낮", "낮은", "flat", "로우"]):
        height_pref = "flat"
    elif any(k in a4 for k in ["굽", "높", "두툼", "mid", "하이"]):
        height_pref = "mid"
    else:
        height_pref = "flat"

    return {
        "weight_pref": weight_pref,
        "vent_pref": vent_pref,
        "cushion_pref": cushion_pref,
        "height_pref": height_pref,
    }

# ====== 점수 함수 (CSV: stability/breathability/cushioning만 사용) ======
def score_row_simple(row, pref):
    # 무게 정보가 없으므로 stability로 안정감, breathability로 통풍, cushioning으로 쿠션만 반영
    # WeightScore(대신 stability 사용)
    if pref["weight_pref"] == "light":
        weight_score = 0.6  # 경량 선호지만 데이터엔 무게 없음 → 낮은 가중치로 기본 가점
        # 카테고리가 tempo/race면 약간 더 가점
        if str(row.category).lower() in ["tempo", "race"]:
            weight_score += 0.4
    elif pref["weight_pref"] == "stable":
        weight_score = 1.0 if row.stability == 2 else (0.5 if row.stability == 1 else 0.0)
    else:  # balanced
        weight_score = 0.5 * (1.0 if row.stability == 2 else (0.5 if row.stability == 1 else 0.0)) + 0.25

    # VentScore
    b = int(row.breathability)
    if pref["vent_pref"] == "high":
        vent_score = {2:1.0, 1:0.6, 0:0.2}[b]
    elif pref["vent_pref"] == "medium":
        vent_score = {2:0.8, 1:0.6, 0:0.4}[b]
    else:  # low
        vent_score = {2:0.3, 1:0.5, 0:1.0}[b]

    # CushionScore
    c = int(row.cushioning)
    if pref["cushion_pref"] == "high":
        cushion_score = {2:1.0, 1:0.6, 0:0.0}[c]
    elif pref["cushion_pref"] == "medium":
        cushion_score = {2:0.8, 1:0.6, 0:0.4}[c]
    else:  # low
        cushion_score = {0:1.0, 1:0.5, 2:0.0}[c]

    # HeightScore (데이터엔 없음 → 카테고리로 근사 가점)
    if pref["height_pref"] == "flat":
        height_score = 0.7 if str(row.category).lower() in ["tempo", "race"] else 0.5
    else:  # mid
        height_score = 0.7 if str(row.category).lower() in ["daily", "stability"] else 0.5

    total = weight_score + vent_score + cushion_score + height_score
    return float(total), float(weight_score), float(vent_score), float(cushion_score), float(height_score)

def pick_top3(catalog_df, pref, seed="0"):
    scored = []
    for r in catalog_df.itertuples(index=False):
        total, ws, vs, cs, hs = score_row_simple(r, pref)
        scored.append((total, r.id))
    scored.sort(key=lambda x: x[0], reverse=True)
    # 동점/순서효과 완화: 상위 6 후보 섞고 3개 선택
    top_ids = [sid for _, sid in scored[:6]]
    rng = random.Random(int(hashlib.sha1(str(seed).encode()).hexdigest(),16)%10**8)
    rng.shuffle(top_ids)
    return top_ids[:3]

def format_price(p):
    try:
        p = int(str(p).replace(",", "").strip())
        return f"{p:,}원"
    except:
        return "가격 업데이트 필요"

def build_recommendation_lines(df, pref, user_reasons_text=""):
    lines = []
    for i, r in enumerate(df.itertuples(index=False), start=1):
        price_txt = format_price(r.price_krw)
        # 설명: 사용자 선호 키워드 요약 반영
        desc_bits = []
        if pref["weight_pref"] == "light":
            desc_bits.append("가벼운 착화감")
        elif pref["weight_pref"] == "stable":
            desc_bits.append("안정감 있는 지지")
        if pref["vent_pref"] == "high":
            desc_bits.append("통풍감 강화")
        if pref["cushion_pref"] == "high":
            desc_bits.append("충격 완화 쿠션")
        if str(r.category).lower() in ["race","tempo"]:
            desc_bits.append("스피드 지향")
        desc = " · ".join(desc_bits) if desc_bits else (r.notes or "").strip()
        if user_reasons_text:
            desc = f"{desc} — {user_reasons_text}"
        # 링크 포함(규칙 준수)
        line = f"- {i}. {r.brand} {r.model} {price_txt} - {desc} (링크: {r.url})"
        lines.append(line)
    return "\n".join(lines)

# ====== 채팅 입력 ======
if user_input := st.chat_input("메시지를 입력해 주세요"):
    # 사용자가 방금 입력하기 직전의 '마지막 어시스턴트 질문'을 보고 Q1~Q4 매핑
    if st.session_state["last_question_tag"] in {"q1","q2","q3","q4"}:
        st.session_state["qa"][st.session_state["last_question_tag"]] = user_input

    st.chat_message("user").write(f"{user_input}")
    st.session_state["messages"].append(("user", user_input))

    # AI 답변
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = ChatOpenAI(model_name="gpt-4o", streaming=True, callbacks=[stream_handler])

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """# 작업 설명: 운동화 쇼핑 에이전트
(이하 동일; 너의 원본 프롬프트 그대로)
"""),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )
        chain = prompt | llm
        chain_with_memory = RunnableWithMessageHistory(
            chain, get_session_history, input_messages_key="question", history_messages_key="history"
        )

        response = chain_with_memory.invoke(
            {"question": user_input},
            config={"configurable": {"session_id": "abc123"}},
        )
    msg = response.content

    # ====== 어시스턴트가 방금 낸 질문을 감지해 '다음 사용자 답변'을 Q에 매핑하도록 태깅 ======
    last_tag = None
    if Q1_TEXT in msg:
        last_tag = "q1"
    elif Q2_TEXT in msg:
        last_tag = "q2"
    elif Q3_TEXT in msg:
        last_tag = "q3"
    elif Q4_TEXT in msg:
        last_tag = "q4"
    st.session_state["last_question_tag"] = last_tag

    # ====== 5단계 시작 문장을 감지하면 CSV 기반 추천으로 덮어쓰기 ======
    if msg.strip().startswith(START_SENTENCE):
        # 1) 사용자 선호 파싱
        pref = parse_preferences_from_answers(st.session_state["qa"])

        # 2) 점수화 & Top-3 선정(시드=세션 고유값)
        pid = st.session_state.get("pid") or str(int(time.time() * 1000))
        st.session_state["pid"] = pid
        top_ids = pick_top3(catalog, pref, seed=pid)
        recs = catalog[catalog["id"].isin(top_ids)]

        # 3) 사용자 이유 텍스트 요약(간단 연결)
        reasons = []
        for k in ["q1","q2","q3","q4"]:
            if st.session_state["qa"][k]:
                reasons.append(st.session_state["qa"][k])
        reasons_text = " / ".join(reasons[:2])  # 너무 길면 2개만

        # 4) 포맷에 맞게 재구성
        lines = build_recommendation_lines(recs, pref, user_reasons_text="")
        msg = START_SENTENCE + "\n\n" + lines

        # (선택) 6단계 종료 멘트가 이어졌다면 유지하고 싶으면 여기서 덧붙여도 됨.

        # (선택) 로그 남기려면 st.session_state에 저장
        st.session_state["last_reco"] = {
            "catalog_version": CATALOG_VER,
            "catalog_hash": catalog_hash,
            "rule_version": RULE_VER,
            "pref": pref,
            "top_ids": top_ids,
            "answers": st.session_state["qa"].copy()
        }

    st.session_state["messages"].append(("assistant", msg))
    st.chat_message("assistant").write(msg)
