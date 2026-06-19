"""
AI 기업 계약 및 사업계획 리스크 분석기 (Business Safety Index, BSI)
------------------------------------------------------------------
- LLM(google-genai / gemini-2.5-flash)은 '데이터 추출기'로만 사용.
- 최종 평가는 학생이 직접 설계한 결정론적 로직으로 산출.
  · calculate_safety_index(): 가중치 + 페널티 기반 안전 지수
  · detect_red_flags(): 독소조항을 규칙으로 직접 탐지 (부정어 가드 포함)
"""

import streamlit as st
from google import genai
from google.genai import types
import plotly.express as px
import pandas as pd
import json
import re

# st.set_page_config는 반드시 최상단에 배치합니다.
st.set_page_config(page_title="AI 기업 계약 리스크 분석기", layout="wide")

# ====== 운영 안전장치 (남용·요금 방지) ======
MAX_ANALYSES_PER_SESSION = 20   # 세션당 분석 횟수 제한
MAX_INPUT_CHARS = 6000          # 입력 길이 제한

# =========================================================
#  1. 결정론적 위험 조항 탐지 (LLM과 무관한, 학생이 직접 설계한 규칙)
# =========================================================
RED_FLAG_RULES = [
    (r"무한\s*책임|무제한\s*책임",            7, "무한·무제한 책임 부담"),
    (r"위약벌",                               6, "위약벌(손해배상과 별도로 부과)"),
    (r"일방적(으로)?\s*(해지|해제|변경|중단)",   6, "상대방의 일방적 해지·변경 권한"),
    (r"손해배상.{0,6}(제한|면제|배제)",          5, "손해배상 책임의 제한·면제"),
    (r"(권리|지위).{0,4}양도.{0,4}금지",         3, "권리 양도 금지(출구 제약)"),
    (r"자동\s*(갱신|연장)",                     2, "자동 갱신·연장 조항"),
    (r"독점|배타적|경업\s*금지",                3, "독점·배타·경업금지 조건"),
    (r"(준거법|관할).{0,10}(외국|해외|영문)",     4, "해외 준거법·관할 지정"),
]
KEYWORD_PENALTY_CAP = 18  # 키워드 페널티 총합 상한 (변별력 보호)

# 부정어 단서: 키워드 직후 이 표현이 나오면 '보호 조항'으로 보고 감점하지 않음.
NEGATION_CUES = ["아니", "않", "없", "불가", "못한", "면한다", "지 아니"]
NEGATION_WINDOW = 25


def _has_negation_near(text: str, end_idx: int) -> bool:
    segment = text[end_idx:end_idx + NEGATION_WINDOW]
    return any(cue in segment for cue in NEGATION_CUES)


def detect_red_flags(text: str):
    """입력 텍스트에서 위험 조항을 직접 탐지한다. (LLM 호출 없음, 부정어 가드 적용)"""
    flags = []
    penalty = 0
    for pattern, pts, label in RED_FLAG_RULES:
        matched = False
        for m in re.finditer(pattern, text):
            if _has_negation_near(text, m.end()):
                continue
            matched = True
            break
        if matched:
            flags.append({"label": label, "penalty": pts})
            penalty += pts
    penalty = min(penalty, KEYWORD_PENALTY_CAP)
    return flags, penalty


# =========================================================
#  2. 평가 알고리즘 (독자적 로직)
# =========================================================
def calculate_safety_index(llm_data: dict, keyword_penalty: int = 0) -> dict:
    legal = llm_data.get('legal_risk', 5)
    business = llm_data.get('business_risk', 5)
    clarity = llm_data.get('clarity', 7)

    weighted_risk = (legal * 0.65) + (business * 0.35)
    base_score = weighted_risk * 7

    clarity_penalty = max(0, (6 - clarity) * 4) if clarity < 6 else 0
    critical_penalty = 25 if legal >= 8 else (15 if legal >= 6 else 0)

    final_score = 100 - base_score - clarity_penalty - critical_penalty - keyword_penalty
    final_score = max(0, min(100, round(final_score, 1)))

    if final_score >= 80:
        level = "🟢 안전 (Safe)"
    elif final_score >= 60:
        level = "🟡 주의 (Caution)"
    else:
        level = "🔴 고위험 (High Risk)"

    return {
        "final_score": final_score, "risk_level": level,
        "legal_risk": legal, "business_risk": business,
        "clarity": clarity, "keyword_penalty": keyword_penalty,
    }


# =========================================================
#  3. Streamlit UI 구성
# =========================================================
st.title("⚖️ AI 기업 계약 및 사업계획 리스크 분석기")
st.markdown("**Business Safety Index (BSI)** 측정 도구")

if "analysis_count" not in st.session_state:
    st.session_state.analysis_count = 0

try:
    default_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    default_key = ""

with st.sidebar:
    st.header("⚙️ 설정")
    user_key = st.text_input("Gemini API Key (선택)", type="password")
    if default_key:
        st.caption("※ 서버 등록 키로 바로 체험할 수 있습니다. 직접 입력하면 그 키를 우선 사용합니다.")
    else:
        st.caption("※ 본인 Gemini API Key를 입력하세요. (로컬 실행 시 필수)")
    st.markdown("---")
    st.caption(f"세션당 분석 한도: {MAX_ANALYSES_PER_SESSION}회 "
               f"(현재 {st.session_state.analysis_count}회 사용)")
    st.info("이 서비스는 LLM을 활용한 보조 분석 도구이며, 법적 효력이 없습니다.")

api_key = user_key.strip() or default_key

with st.expander("🧪 채점 로직 검증 결과 보기 (샘플 4종 · API 불필요)"):
    demo_cases = [
        ("우량 계약", {"legal_risk": 1, "business_risk": 2, "clarity": 9},
         "갑과 을은 상호 협의하여 계약을 갱신한다."),
        ("평범 계약", {"legal_risk": 4, "business_risk": 4, "clarity": 7},
         "본 계약은 1년간 유효하며 자동 갱신된다."),
        ("모호 계약", {"legal_risk": 5, "business_risk": 5, "clarity": 3},
         "적절한 시점에 합리적 수준으로 정산한다."),
        ("독소 계약", {"legal_risk": 9, "business_risk": 7, "clarity": 4},
         "을은 무한 책임을 지며, 갑은 일방적으로 해지할 수 있고 위약벌을 부과한다."),
    ]
    rows = []
    for name, llm, txt in demo_cases:
        _, kp = detect_red_flags(txt)
        r = calculate_safety_index(llm, kp)
        rows.append({
            "시나리오": name,
            "법/비즈/명확": f"{llm['legal_risk']}/{llm['business_risk']}/{llm['clarity']}",
            "독소 감점": -kp,
            "BSI": r["final_score"],
            "등급": r["risk_level"],
        })
    st.table(pd.DataFrame(rows))
    st.caption("위험도가 높아질수록 BSI가 단조 감소하도록 설계되었습니다.")

text_input = st.text_area("분석할 계약서 조항이나 사업 계획을 입력하세요:", height=200)

if st.button("🔍 리스크 분석 실행", type="primary"):
    if not api_key:
        st.error("좌측 사이드바에서 Gemini API Key를 먼저 입력해주세요!")
    elif not text_input.strip():
        st.warning("분석할 텍스트를 입력해주세요.")
    elif st.session_state.analysis_count >= MAX_ANALYSES_PER_SESSION:
        st.error(f"이 세션의 분석 한도({MAX_ANALYSES_PER_SESSION}회)에 도달했습니다. "
                 "페이지를 새로고침하면 초기화됩니다.")
    else:
        if len(text_input) > MAX_INPUT_CHARS:
            st.warning(f"입력이 너무 깁니다. 앞 {MAX_INPUT_CHARS}자만 분석합니다.")
        target_text = text_input[:MAX_INPUT_CHARS]

        try:
            client = genai.Client(api_key=api_key)

            prompt = f"""
            너는 기업 계약서와 사업 계획서를 분석하는 전문 리스크 분석가다.
            아래 텍스트를 분석하여 Legal Risk(0~10), Business Risk(0~10), Clarity(0~10)를 평가하라.
            법적 리스크 평가는 보수적으로(Conservative) 평가하라.

            반드시 아래 JSON 형식으로만 출력하라:
            {{
              "legal_risk": 정수,
              "business_risk": 정수,
              "clarity": 정수,
              "key_risks": ["위험1", "위험2"],
              "summary": "1~2문장 요약"
            }}

            분석할 텍스트:
            {target_text}
            """

            with st.spinner("AI가 법률 및 비즈니스 리스크를 분석 중입니다... ⏳"):
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0,
                    ),
                )
                try:
                    llm_result = json.loads(response.text)
                except (json.JSONDecodeError, TypeError):
                    st.error("AI가 올바른 JSON 형식으로 응답하지 않았습니다. 다시 시도해주세요.")
                    st.stop()

            st.session_state.analysis_count += 1

            red_flags, keyword_penalty = detect_red_flags(target_text)
            eval_result = calculate_safety_index(llm_result, keyword_penalty)

            st.divider()
            st.subheader("📊 분석 결과 대시보드")
            col1, col2 = st.columns([1, 2])

            with col1:
                st.metric(label="최종 안전 지수 (BSI)", value=f"{eval_result['final_score']} 점")
                st.metric(label="위험 등급", value=eval_result['risk_level'])
                if keyword_penalty > 0:
                    st.metric(label="독소조항 가산 감점", value=f"-{keyword_penalty} 점")
                st.markdown("### 📝 핵심 요약")
                st.info(llm_result.get('summary', '요약 없음'))

            with col2:
                df = pd.DataFrame(dict(
                    r=[eval_result['legal_risk'], eval_result['business_risk'], eval_result['clarity']],
                    theta=['법적 리스크 (낮을수록 좋음)', '비즈니스 리스크 (낮을수록 좋음)', '명확성 (높을수록 좋음)']
                ))
                fig = px.line_polar(df, r='r', theta='theta', line_close=True, range_r=[0, 10])
                fig.update_traces(fill='toself')
                st.plotly_chart(fig, use_container_width=True)

            if red_flags:
                st.markdown("### 🚨 직접 탐지한 독소조항 (규칙 기반)")
                for f in red_flags:
                    st.warning(f"- {f['label']}  (감점 {f['penalty']}점)")

            st.markdown("### ⚠️ 주요 위험 요소 (AI 분석)")
            for risk in llm_result.get('key_risks', []):
                st.error(f"- {risk}")

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
            st.error("API 키가 올바른지, 네트워크 상태가 정상인지 확인 후 다시 시도해주세요.")

st.divider()
st.caption("⚠️ 면책 조항: 본 서비스의 분석 결과는 참고용이며 법적 효력이 없습니다. "
           "규칙 기반 탐지는 보수적으로 설계되었으며 문맥을 완전히 이해하지 못할 수 있습니다. "
           "최종 의사결정은 반드시 전문 변호사 및 회계사와 상담하십시오.")
