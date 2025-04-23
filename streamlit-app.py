import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3


# 페이지 설정
st.set_page_config(
   page_title="트레이딩 모니터링 대시보드",
   page_icon="📊",
   layout="wide"
)


# 데이터베이스 연결
@st.cache_resource
def get_database_connection():
   return sqlite3.connect('trading.db', check_same_thread=False)


conn = get_database_connection()


# 데이터 로드 함수들
@st.cache_data(ttl=60)  # 1분마다 데이터 갱신
def load_recent_trades():
   query = """
       SELECT
           timestamp,
           decision,
           percentage,
           reason,
           btc_balance,
           krw_balance,
           btc_avg_buy_price,
           btc_krw_price
       FROM trading_history
       ORDER BY timestamp DESC
   """
   return pd.read_sql_query(query, conn)


@st.cache_data(ttl=60)
def load_reflections():
   query = """
       SELECT
           r.*,
           h.decision,
           h.percentage,
           h.btc_krw_price
       FROM trading_reflection r
       JOIN trading_history h ON r.trading_id = h.id
       ORDER BY r.reflection_date DESC
   """
   return pd.read_sql_query(query, conn)


# 메인 대시보드
st.title("📊 트레이딩 모니터링 대시보드")


# 데이터 로드
trades_df = load_recent_trades()
reflections_df = load_reflections()


# 상단 메트릭스
col1, col2, col3, col4 = st.columns(4)


with col1:
   latest_btc_price = trades_df.iloc[0]['btc_krw_price'] if not trades_df.empty else 0
   st.metric("현재 BTC 가격", f"{latest_btc_price:,.0f} KRW")


with col2:
   latest_btc_balance = trades_df.iloc[0]['btc_balance'] if not trades_df.empty else 0
   st.metric("BTC 보유량", f"{latest_btc_balance:.8f} BTC")


with col3:
   latest_krw_balance = trades_df.iloc[0]['krw_balance'] if not trades_df.empty else 0
   st.metric("KRW 잔고", f"{latest_krw_balance:,.0f} KRW")


with col4:
   total_value = latest_btc_balance * latest_btc_price + latest_krw_balance
   st.metric("총 자산가치", f"{total_value:,.0f} KRW")


# 차트 섹션
st.subheader("📈 거래 히스토리")
trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])


# BTC 가격 차트
fig_price = px.line(trades_df,
                   x='timestamp',
                   y='btc_krw_price',
                   title='BTC 가격 변동')
fig_price.update_layout(height=400)
st.plotly_chart(fig_price, use_container_width=True)


# 매수/매도 결정 분석
col1, col2 = st.columns(2)


with col1:
   decision_counts = trades_df['decision'].value_counts()
   fig_decisions = px.pie(values=decision_counts.values,
                         names=decision_counts.index,
                         title='매수/매도 비율')
   st.plotly_chart(fig_decisions)


with col2:
   avg_percentage_by_decision = trades_df.groupby('decision')['percentage'].mean()
   fig_percentages = px.bar(x=avg_percentage_by_decision.index,
                           y=avg_percentage_by_decision.values,
                           title='결정별 평균 변동률')
   st.plotly_chart(fig_percentages)


# 최근 거래 내역 테이블
st.subheader("📝 최근 거래 내역")
recent_trades = trades_df[['timestamp', 'decision', 'percentage', 'reason', 'btc_krw_price']].head(10)
recent_trades['timestamp'] = recent_trades['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
recent_trades.columns = ['시간', '결정', '변동률', '사유', 'BTC 가격']
st.dataframe(recent_trades, use_container_width=True)


# 반성일기 섹션
st.subheader("📔 트레이딩 반성일기")
if not reflections_df.empty:
   reflections_df['reflection_date'] = pd.to_datetime(reflections_df['reflection_date'])
   recent_reflections = reflections_df.head(5)
  
   for _, reflection in recent_reflections.iterrows():
       with st.expander(f"반성일기 - {reflection['reflection_date'].strftime('%Y-%m-%d')}"):
           col1, col2 = st.columns(2)
           with col1:
               st.write("**시장 상황:**", reflection['market_condition'])
               st.write("**의사결정 분석:**", reflection['decision_analysis'])
           with col2:
               st.write("**개선점:**", reflection['improvement_points'])
               st.write("**성공률:**", f"{reflection['success_rate']:.1f}%")
           st.write("**학습 포인트:**", reflection['learning_points'])


# 사이드바에 필터 추가
st.sidebar.title("📊 필터 옵션")
date_range = st.sidebar.date_input(
   "날짜 범위 선택",
   value=(datetime.now() - timedelta(days=7), datetime.now())
)


decision_filter = st.sidebar.multiselect(
   "거래 유형",
   options=trades_df['decision'].unique(),
   default=trades_df['decision'].unique()
)


# 자동 새로고침 옵션
st.sidebar.write("---")
auto_refresh = st.sidebar.checkbox("자동 새로고침", value=True)
if auto_refresh:
   refresh_interval = st.sidebar.slider("새로고침 간격(초)",
                                      min_value=5,
                                      max_value=300,
                                      value=60)
   st.empty()



