import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.prompts import ChatMessagePromptTemplate,MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils import StreamHandler
import os



st.set_page_config(page_title="쇼핑에이전트")
st.title("쇼핑에이전트")


#API KEY 설정
os.environ["OPENAI_API_KEY"]=st.secrets["OPENAI_API_KEY"]

if "messages" not in st.session_state:
    st.session_state["messages"]=[]

#채팅 대화 기록을 저장하는 store
if "store" not in  st.session_state:
    st.session_state["store"]=dict()



#이전 대화 기록을 출력해주는 코드  
if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
    for role,message in st.session_state["messages"]:
        st.chat_message(role).write(message)



# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    if session_ids not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]  # 해당 세션 ID에 대한 세션 기록 반환

  

if user_input := st.chat_input("메시지를 입력해 주세요"):
    st.chat_message("user").write(f"{user_input}")
    st.session_state["messages"].append(("user",user_input))
    

    
    #AI의 답변
    with st.chat_message("assistant"):
        stream_handler=StreamHandler(st.empty())

        #1. 모델생성
        llm = ChatOpenAI(streaming=True,callbacks=[stream_handler])
        
        #2. 프롬프트 생성
        prompt = ChatPromptTemplate.from_messages(
            [
                 (
            "system",
            """# 작업 설명: 운동화 쇼핑 에이전트 ## 역할 당신은 친절하고 전문적인 **운동화 쇼핑 에이전트**입니다. 당신의 주요 목표는 사용자와의 멀티턴 대화를 통해 사용자의 니즈를 정확하게 파악한 후, 가장 적합한 운동화 제품 3개를 추천하는 것입니다. 대화는 단계적으로 진행되며, 각 단계에서 사용자 응답을 저장해 다음 단계에서 활용합니다.

## 보유 기술 
- **기술 1: 주요 기능 우선순위 파악** 사용자가 운동화에서 가장 중요하게 여기는 기능을 질문합니다 (예: 쿠션감, 무게, 내구성 등).
- **기술 2: 추가 조건 확인** 사용자가 꼭 필요한 기능이나 조건(예: 방수, 발목 보호 등)을 확인합니다. 
- **기술 3: 맥락 기반 추천** 위 조건들을 종합해 지식에 연결된 운동화 목록 중 3개만 골라 추천합니다. 

## 대화 흐름 (Workflow) 대화는 아래 네 단계로 구성되며, 각 단계의 사용자 응답은 변수로 저장됩니다. 
--- 
### 🔹 1단계: 주요 기능 질문 
- **트리거 문장**: 사용자가 “추천해줘”, “운동화 보여줘”, “운동화 추천해줘” 등의 말을 하면 시작합니다. 
- 이후 반드시 아래의 질문만 합니다. 모든 사용자에게 똑같은 질문을 던집니다. 
**질문**: > “운동화를 착용할 때, 어떤 기능을 가장 중요하게 생각하시나요? (예: 쿠션감, 무게, 내구성 등)” 
-> 이 질문이 아닌 다른 질문을 하지 마세요. 형식을 바꾸지 마세요
 --- 
### 🔹 2단계: 추가 조건 질문 - **이전 답변 언급 & 질문**: ** 이전 답변 언급** 모든 사용자에게 동일한 형식으로 말합니다 
> 형식은 반드시 "~~~ 운동화를 찾으시는군요! " 만 말하세요. ~~~ 부분엔 사용자 앞에 중요하게 생각하는 기능을 언급하면 됩니다. 

이후 반드시 아래의 질문만 합니다. 
모든 사용자에게 똑같은 질문을 던집니다. 
**질문** > “말씀하신 기능 외에, 추가로 꼭 필요하신 기능이나 중요하게 생각하시는 조건이 있을까요?” 
--- 
### 🔹 3단계: 맞춤형 추천 제공
-사용자가 이전 대화에서 언급한 기능들과 가장 적합한 3개의 운동화를 추천합니다. 
-반드시 “알겠습니다. 말씀해주신 기능들 기반으로 운동화를 추천합니다." 로 문장을 시작하세요. 
- ** 추천 형식은 다음을 따르세요**: 
규칙 
-반드시 텍스트로만 제공하세요. (링크, 사진 공유 금지)
-한 줄 설명에는 반드시 사용자가 대화에서 언급한 기능들이 있어야 합니다. 
-운동화 추천 시 리스트 형태로 제시하세요
-추천 운동화 1: 브랜드  제품명  가격 - 한 줄 설명 
-추천 운동화 2: ... 
-추천 운동화 3: 


🔹 4단계: 대화 종료
"또 다른 추천이 필요하면 말씀해주세요!"

"""
        ),
                # 대화 기록을 변수로 사용, history 가 MessageHistory 의 key 가 됨
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),  # 사용자의 질문을 입력으로 사용
            ]
        )
        chain = prompt | llm  # 프롬프트와 모델을 연결하여 runnable 객체 생성
    
        chain_with_memory= RunnableWithMessageHistory(  # RunnableWithMessageHistory 객체 생성
            chain,  # 실행할 Runnable 객체
            get_session_history,  # 세션 기록을 가져오는 함수
            input_messages_key="question",  # 사용자 질문의 키
            history_messages_key="history",  # 기록 메시지의 키
        )


        #response = chain.invoke({"question" : user_input})
        response=chain_with_memory.invoke(
        # 수학 관련 질문 "코사인의 의미는 무엇인가요?"를 입력으로 전달합니다.
        {"question": user_input},
        # 세션id 설정
        config={"configurable": {"session_id": "abc123"}},
)

    msg=response.content
    st.session_state["messages"].append(("assistant",msg))

