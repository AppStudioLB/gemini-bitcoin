import os
from dotenv import load_dotenv
import json
import pyupbit
import pandas as pd
import ta
from datetime import datetime, timedelta
from openai import OpenAI
import time
import requests
import base64
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from selenium.webdriver.chrome.service import Service

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from PIL import Image
import io
from youtube_transcript_api import YouTubeTranscriptApi

import sqlite3  # SQLite 추가

def is_ec2():
    """EC2 환경인지 확인"""
    try:
        return os.path.exists("/sys/hypervisor/uuid")
    except:
        return False


def setup_chrome_options():
    """Chrome 옵션 설정"""
    chrome_options = Options()
   
    # 공통 옵션
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
   
    if is_ec2():
        # EC2 환경 전용 옵션
        chrome_options.add_argument("--headless")
    else:
        # 로컬 환경 전용 옵션
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
        if os.getenv('HEADLESS', 'false').lower() == 'true':
            chrome_options.add_argument("--headless")
   
    return chrome_options


def create_driver():
    """WebDriver 생성"""
    try:
        env_type = 'EC2' if is_ec2() else '로컬'
        print(f"{env_type} 환경에서 ChromeDriver 설정 중...")
       
        chrome_options = setup_chrome_options()
       
        if is_ec2():
            service = Service('/usr/bin/chromedriver')
        else:
            try:
                from webdriver_manager.chrome import ChromeDriverManager
                service = Service(ChromeDriverManager().install())
            except ImportError:
                service = Service('chromedriver')
       
        return webdriver.Chrome(service=service, options=chrome_options)
       
    except Exception as e:
        print(f"ChromeDriver 생성 중 오류 발생: {e}")
        raise

# DatabaseManager 클래스 수정
class DatabaseManager:
    def __init__(self, db_path="trading.db"):
        self.conn = sqlite3.connect(db_path)
        self.setup_database()
       
    def setup_database(self):
        cursor = self.conn.cursor()
        # 기존 거래 기록 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                decision TEXT NOT NULL,
                percentage REAL NOT NULL,
                reason TEXT NOT NULL,
                btc_balance REAL NOT NULL,
                krw_balance REAL NOT NULL,
                btc_avg_buy_price REAL NOT NULL,
                btc_krw_price REAL NOT NULL
            )
        """)
       
        # 거래 반성 일기 테이블 추가
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_reflection (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trading_id INTEGER NOT NULL,
                reflection_date DATETIME NOT NULL,
                market_condition TEXT NOT NULL,
                decision_analysis TEXT NOT NULL,
                improvement_points TEXT NOT NULL,
                success_rate REAL NOT NULL,
                learning_points TEXT NOT NULL,
                FOREIGN KEY (trading_id) REFERENCES trading_history(id)
            )
        """)
        self.conn.commit()


    def get_recent_trades(self, limit=10):
        """최근 거래 내역 조회"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM trading_history
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        return cursor.fetchall()


    def get_reflection_history(self, limit=10):
        """최근 반성 일기 조회"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT r.*, h.decision, h.percentage, h.btc_krw_price
            FROM trading_reflection r
            JOIN trading_history h ON r.trading_id = h.id
            ORDER BY r.reflection_date DESC
            LIMIT ?
        """, (limit,))
        return cursor.fetchall()


    def add_reflection(self, reflection_data):
        """반성 일기 추가"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO trading_reflection (
                trading_id, reflection_date, market_condition,
                decision_analysis, improvement_points, success_rate,
                learning_points
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            reflection_data['trading_id'],
            reflection_data['reflection_date'],
            reflection_data['market_condition'],
            reflection_data['decision_analysis'],
            reflection_data['improvement_points'],
            reflection_data['success_rate'],
            reflection_data['learning_points']
        ))
        self.conn.commit()


    def record_trade(self, trade_data):
        """거래 데이터를 데이터베이스에 기록"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO trading_history (
                timestamp, decision, percentage, reason,
                btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(),
            trade_data['decision'],
            trade_data['percentage'],
            trade_data['reason'],
            trade_data['btc_balance'],
            trade_data['krw_balance'],
            trade_data['btc_avg_buy_price'],
            trade_data['btc_krw_price']
        ))
        self.conn.commit()
        return cursor.lastrowid  # 새로 삽입된 레코드의 ID 반환






class TradeManager:
    """거래 실행을 담당하는 클래스"""
    def __init__(self, upbit_client, ticker="KRW-BTC"):
        self.upbit = upbit_client
        self.ticker = ticker
        self.MIN_TRADE_AMOUNT = 5000


    def execute_market_buy(self, amount):
        """시장가 매수 주문 실행"""
        if amount >= self.MIN_TRADE_AMOUNT:
            return self.upbit.buy_market_order(self.ticker, amount)
        return None


    def execute_market_sell(self, amount):
        """시장가 매도 주문 실행"""
        current_price = float(pyupbit.get_current_price(self.ticker))
        if amount * current_price >= self.MIN_TRADE_AMOUNT:
            return self.upbit.sell_market_order(self.ticker, amount)
        return None


    def adjust_trade_ratio(self, base_ratio, fear_greed_value, trade_type):
        """공포탐욕지수에 따른 거래 비율 조정"""
        trade_ratio = base_ratio / 100.0
       
        if trade_type == "buy":
            if fear_greed_value <= 25:
                trade_ratio = min(trade_ratio * 1.2, 1.0)
            elif fear_greed_value >= 75:
                trade_ratio = trade_ratio * 0.8
        elif trade_type == "sell":
            if fear_greed_value >= 75:
                trade_ratio = min(trade_ratio * 1.2, 1.0)
            elif fear_greed_value <= 25:
                trade_ratio = trade_ratio * 0.8
               
        return trade_ratio


    def get_current_balances(self):
        """현재 잔고 상태 조회"""
        return {
            'btc_balance': float(self.upbit.get_balance(self.ticker)),
            'krw_balance': float(self.upbit.get_balance("KRW")),
            'btc_avg_buy_price': float(self.upbit.get_avg_buy_price(self.ticker)),
            'btc_krw_price': float(pyupbit.get_current_price(self.ticker))
        }


load_dotenv()




def capture_full_page(url, output_path):
    """웹페이지 캡처 함수"""
    driver = None
    try:
        driver = create_driver()
        wait = WebDriverWait(driver, 20)
       
        driver.get(url)
        time.sleep(5)  # 초기 로딩 대기
       
        try:
            # 시간 설정 버튼 클릭
            time_button = wait.until(EC.element_to_be_clickable((By.XPATH,
                "/html/body/div[1]/div[2]/div[3]/div/section[1]/article[1]/div/span[2]/div/div/div[1]/div[1]/div/cq-menu[1]/span/cq-clickable")))
            time_button.click()
            time.sleep(1)
           
            # 1시간 옵션 클릭
            hour_option = wait.until(EC.element_to_be_clickable((By.XPATH,
                "/html/body/div[1]/div[2]/div[3]/div/section[1]/article[1]/div/span[2]/div/div/div[1]/div[1]/div/cq-menu[1]/cq-menu-dropdown/cq-item[8]")))
            hour_option.click()
            time.sleep(3)
        except TimeoutException:
            print("차트 시간 설정을 찾을 수 없습니다. 기본 설정으로 진행합니다.")
       
        # 전체 페이지 높이 구하기
        total_height = driver.execute_script("return document.body.scrollHeight")
        driver.set_window_size(1920, total_height)
       
        # 스크린샷 캡처
        png = driver.get_screenshot_as_png()
       
        # PIL Image로 변환 및 최적화
        img = Image.open(io.BytesIO(png))
        img.thumbnail((2000, 2000))
        img.save(output_path, optimize=True, quality=85)
        print(f"차트 이미지 저장 완료: {output_path}")
        return True
       
    except Exception as e:
        print(f"페이지 캡처 중 오류 발생: {e}")
        return False
       
    finally:
        if driver:
            driver.quit()








class EnhancedCryptoTrader:
    def __init__(self, ticker="KRW-BTC"):
        self.ticker = ticker
        self.access = os.getenv('UPBIT_ACCESS_KEY')
        self.secret = os.getenv('UPBIT_SECRET_KEY')
        self.upbit = pyupbit.Upbit(self.access, self.secret)


        # 하위 매니저 클래스들 초기화
        self.trade_manager = TradeManager(self.upbit, ticker)
        self.db = DatabaseManager()


        # 기타 설정
        self.client = OpenAI()
        self.serpapi_key = os.getenv('SERPAPI_KEY')
        self.fear_greed_api = "https://api.alternative.me/fng/"
        self.youtube_channels = ["3XbtEX3jUv4"]




    def analyze_past_decisions(self):
        """과거 거래 분석 및 반성"""
        print("analyze_past_decisions")
        try:
            # 최근 거래 내역 조회
            recent_trades = self.db.get_recent_trades(10)
            recent_reflections = self.db.get_reflection_history(5)
           
            # 현재 시장 상태 조회
            current_market = {
                "price": float(pyupbit.get_current_price(self.ticker)),
                "status": self.get_current_status(),
                "fear_greed": self.get_fear_greed_index(),
                "technical": self.get_ohlcv_data()
            }
            print("1")
            # AI에 분석 요청
            reflection_prompt = {
                "recent_trades": recent_trades,
                "recent_reflections": recent_reflections,
                "current_market": current_market
            }


            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                     {
                        "role": "system",
                        "content": """You are an AI trading advisor. Provide your analysis in JSON format with these exact fields:
                        {
                            "market_condition": "Current market state analysis",
                            "decision_analysis": "Analysis of past trading decisions",
                            "improvement_points": "Points to improve",
                            "success_rate": numeric value between 0-100,
                            "learning_points": "Key lessons learned"
                        }"""
                    },
                    {
                        "role": "user",
                        "content": f"Analyze these trading records and market conditions and provide response in JSON format:\n{json.dumps(reflection_prompt, indent=2)}"
                    }
                ],
                response_format={
                    "type": "json_object"
                }
            )
            print("2")
            reflection = json.loads(response.choices[0].message.content)
           
            # 반성 일기 저장
            reflection_data = {
                'trading_id': recent_trades[0][0],  # 최근 거래 ID
                'reflection_date': datetime.now(),
                'market_condition': reflection['market_condition'],
                'decision_analysis': reflection['decision_analysis'],
                'improvement_points': reflection['improvement_points'],
                'success_rate': reflection['success_rate'],
                'learning_points': reflection['learning_points']
            }
            print("3")
               
            print("\n=== Reflection Data ===")
            print(json.dumps(reflection_data, indent=2, default=str))  # datetime 객체를 위해 default=str 추가
       
               
            self.db.add_reflection(reflection_data)
           
            return reflection


        except Exception as e:
            print(f"Error in analyze_past_decisions: {e}")
            return None








    def get_fear_greed_index(self, limit=7):
        """공포탐욕지수 데이터 조회"""
        try:
            response = requests.get(f"{self.fear_greed_api}?limit={limit}")
            if response.status_code == 200:
                data = response.json()
               
                latest = data['data'][0]
                print("\n=== Fear and Greed Index ===")
                print(f"Current Value: {latest['value']} ({latest['value_classification']})")
               
                processed_data = []
                for item in data['data']:
                    processed_data.append({
                        'date': datetime.fromtimestamp(int(item['timestamp'])).strftime('%Y-%m-%d'),
                        'value': int(item['value']),
                        'classification': item['value_classification']
                    })
               
                values = [int(item['value']) for item in data['data']]
                avg_value = sum(values) / len(values)
                trend = 'Improving' if values[0] > avg_value else 'Deteriorating'
               
                return {
                    'current': {
                        'value': int(latest['value']),
                        'classification': latest['value_classification']
                    },
                    'history': processed_data,
                    'trend': trend,
                    'average': avg_value
                }
               
            return None
        except Exception as e:
            print(f"Error in get_fear_greed_index: {e}")
            return None








    def add_technical_indicators(self, df):
        """기술적 분석 지표 추가"""
        # 볼린저 밴드
        indicator_bb = ta.volatility.BollingerBands(close=df['close'])
        df['bb_high'] = indicator_bb.bollinger_hband()
        df['bb_mid'] = indicator_bb.bollinger_mavg()
        df['bb_low'] = indicator_bb.bollinger_lband()
        df['bb_pband'] = indicator_bb.bollinger_pband()
       
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close']).rsi()
       
        # MACD
        macd = ta.trend.MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
       
        # 이동평균선
        df['ma5'] = ta.trend.SMAIndicator(close=df['close'], window=5).sma_indicator()
        df['ma20'] = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['ma60'] = ta.trend.SMAIndicator(close=df['close'], window=60).sma_indicator()
        df['ma120'] = ta.trend.SMAIndicator(close=df['close'], window=120).sma_indicator()
       
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(
            high=df['high'], low=df['low'], close=df['close']
        ).average_true_range()
       
        return df








    def get_current_status(self):
        """현재 투자 상태 조회"""
        try:
            krw_balance = float(self.upbit.get_balance("KRW"))
            crypto_balance = float(self.upbit.get_balance(self.ticker))
            avg_buy_price = float(self.upbit.get_avg_buy_price(self.ticker))
            current_price = float(pyupbit.get_current_price(self.ticker))
           
            print("\n=== Current Investment Status ===")
            print(f"보유 현금: {krw_balance:,.0f} KRW")
            print(f"보유 코인: {crypto_balance:.8f} {self.ticker}")
            print(f"평균 매수가: {avg_buy_price:,.0f} KRW")
            print(f"현재가: {current_price:,.0f} KRW")
           
            total_value = krw_balance + (crypto_balance * current_price)
            unrealized_profit = ((current_price - avg_buy_price) * crypto_balance) if crypto_balance else 0
            profit_percentage = ((current_price / avg_buy_price) - 1) * 100 if crypto_balance else 0
           
            print(f"미실현 손익: {unrealized_profit:,.0f} KRW ({profit_percentage:.2f}%)")
           
            return {
                "krw_balance": krw_balance,
                "crypto_balance": crypto_balance,
                "avg_buy_price": avg_buy_price,
                "current_price": current_price,
                "total_value": total_value,
                "unrealized_profit": unrealized_profit,
                "profit_percentage": profit_percentage
            }
        except Exception as e:
            print(f"Error in get_current_status: {e}")
            return None








    def get_orderbook_data(self):
        """호가 데이터 조회"""
        try:
            orderbook = pyupbit.get_orderbook(ticker=self.ticker)
            if not orderbook or len(orderbook) == 0:
                return None
               
            ask_prices = []
            ask_sizes = []
            bid_prices = []
            bid_sizes = []
           
            for unit in orderbook['orderbook_units'][:5]:
                ask_prices.append(unit['ask_price'])
                ask_sizes.append(unit['ask_size'])
                bid_prices.append(unit['bid_price'])
                bid_sizes.append(unit['bid_size'])
           
            return {
                "timestamp": datetime.fromtimestamp(orderbook['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                "total_ask_size": float(orderbook['total_ask_size']),
                "total_bid_size": float(orderbook['total_bid_size']),
                "ask_prices": ask_prices,
                "ask_sizes": ask_sizes,
                "bid_prices": bid_prices,
                "bid_sizes": bid_sizes
            }
        except Exception as e:
            print(f"Error in get_orderbook_data: {e}")
            return None








    def get_ohlcv_data(self):
        """차트 데이터 수집 및 기술적 분석"""
        try:
            daily_data = pyupbit.get_ohlcv(self.ticker, interval="day", count=30)
            daily_data = self.add_technical_indicators(daily_data)
           
            hourly_data = pyupbit.get_ohlcv(self.ticker, interval="minute60", count=24)
            hourly_data = self.add_technical_indicators(hourly_data)
           
            daily_data_dict = []
            for index, row in daily_data.iterrows():
                day_data = row.to_dict()
                day_data['date'] = index.strftime('%Y-%m-%d')
                daily_data_dict.append(day_data)
               
            hourly_data_dict = []
            for index, row in hourly_data.iterrows():
                hour_data = row.to_dict()
                hour_data['date'] = index.strftime('%Y-%m-%d %H:%M:%S')
                hourly_data_dict.append(hour_data)
           
            print("\n=== Latest Technical Indicators ===")
            print(f"RSI: {daily_data['rsi'].iloc[-1]:.2f}")
            print(f"MACD: {daily_data['macd'].iloc[-1]:.2f}")
            print(f"BB Position: {daily_data['bb_pband'].iloc[-1]:.2f}")
           
            return {
                "daily_data": daily_data_dict[-7:],
                "hourly_data": hourly_data_dict[-6:],
                "latest_indicators": {
                    "rsi": daily_data['rsi'].iloc[-1],
                    "macd": daily_data['macd'].iloc[-1],
                    "macd_signal": daily_data['macd_signal'].iloc[-1],
                    "bb_position": daily_data['bb_pband'].iloc[-1]
                }
            }
        except Exception as e:
            print(f"Error in get_ohlcv_data: {e}")
            return None
    def capture_and_analyze_chart(self):
        """차트 캡처 및 분석"""
        try:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"chart_{current_time}.png"
           
            url = f"https://upbit.com/exchange?code=CRIX.UPBIT.{self.ticker}"
            capture_success = capture_full_page(url, screenshot_path)
           
            if not capture_success:
                return None
           
            # 이미지를 base64로 인코딩
            with open(screenshot_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
           
            # OpenAI Vision API 호출
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Analyze this cryptocurrency chart and provide insights about: 1) Current trend 2) Key support/resistance levels 3) Technical indicator signals 4) Notable patterns"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=500
                )
           
            # 분석 결과 처리
            analysis_result = response.choices[0].message.content
           
            # 임시 파일 삭제
            os.remove(screenshot_path)
           
            return analysis_result
           
        except Exception as e:
            print(f"Error in capture_and_analyze_chart: {e}")
            if os.path.exists(screenshot_path):
                os.remove(screenshot_path)
            return None








    def get_crypto_news(self):
        """비트코인 관련 최신 뉴스 조회"""
        try:
            base_url = "https://serpapi.com/search.json"
            params = {
                "engine": "google_news",
                "q": "bitcoin crypto trading",
                "api_key": self.serpapi_key,
                "gl": "us",
                "hl": "en"
            }
           
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                news_data = response.json()
               
                if 'news_results' not in news_data:
                    return None
                   
                processed_news = []
                for news in news_data['news_results'][:5]:
                    processed_news.append({
                        'title': news.get('title', ''),
                        'link': news.get('link', ''),
                        'source': news.get('source', {}).get('name', ''),
                        'date': news.get('date', ''),
                        'snippet': news.get('snippet', '')
                    })
               
                print("\n=== Latest Crypto News ===")
                for news in processed_news:
                    print(f"\nTitle: {news['title']}")
                    print(f"Source: {news['source']}")
                    print(f"Date: {news['date']}")
               
                return processed_news
           
            return None
        except Exception as e:
            print(f"Error in get_crypto_news: {e}")
            return None








    # [이전 코드의 나머지 메서드들은 그대로 유지...]
    # get_fear_greed_index, add_technical_indicators, get_current_status,
    # get_orderbook_data, get_ohlcv_data 메서드들은 변경 없이 유지


    def get_youtube_analysis(self):
        """유튜브 영상 자막 분석"""
        try:
            with open('strategy.txt', 'r', encoding='utf-8') as f:
                content = f.read()
            # all_transcripts = []
       
            # for video_id in self.youtube_channels:
            #     try:
            #         # 한국어 자막 가져오기 시도
            #         transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko'])
            #         text = ' '.join([entry['text'] for entry in transcript])
                   
            #         all_transcripts.append({
            #             'video_id': video_id,
            #             'content': text
            #         })
                       
            #     except Exception as e:
            #         print(f"Error processing transcript for video {video_id}: {e}")
            #         # 사용 가능한 자막 목록 출력
            #         try:
            #             available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
            #             print(f"Available transcripts for {video_id}:")
            #             for transcript in available_transcripts:
            #                 print(f"- {transcript.language_code}")
            #         except Exception as e:
            #             print(f"Could not list available transcripts: {e}")
            #         continue
           
           
            # if not all_transcripts:
            #     return None


            # 한국어 분석을 위한 시스템 메시지
            system_message = """You are an expert cryptocurrency trading analyst.
    Analyze Korean YouTube content related to cryptocurrency trading and provide insights.


    Focus on analyzing these key aspects from the Korean transcripts:
    1. Trading Strategy
    - Entry/exit points
    - Risk management methods
    - Trading patterns


    2. Market Analysis
    - Market sentiment
    - Important price levels
    - Potential scenarios


    3. Risk Factors
    - Market risks
    - Technical risks
    - External risks


    4. Technical Analysis
    - Technical indicators
    - Chart patterns
    - Key price levels


    5. Market Impact Factors
    - Economic factors
    - News and events
    - Market trends


    Provide analysis in JSON format with confidence scores."""


            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Analyze these Korean cryptocurrency trading YouTube transcripts: {content}"}
                ],
                response_format={"type": "json_object"}
            )


            try:
                analysis_result = json.loads(response.choices[0].message.content)
                return analysis_result
            except json.JSONDecodeError as e:
                print(f"JSON Parsing Error: {e}")
                print("Original response:", response.choices[0].message.content)
                return None


        except Exception as e:
            print(f"Error in get_youtube_analysis: {e}")
            return None


    def get_ai_analysis(self, analysis_data):
        """AI 분석 및 매매 신호 생성 (Structured Outputs 적용)"""
        try:
            # 차트 이미지 분석 수행
            chart_analysis = self.capture_and_analyze_chart()
            # 유튜브 분석 수행
            youtube_analysis = self.get_youtube_analysis()
           
            # 과거 반성 일기 분석 추가
            past_reflections = self.db.get_reflection_history(5)
           
            # 분석 데이터 최적화
            optimized_data = {
                "current_status": analysis_data["current_status"],
                "orderbook": analysis_data["orderbook"],
                "ohlcv": analysis_data["ohlcv"],
                "fear_greed": analysis_data["fear_greed"],
                "news": analysis_data["news"],
                "chart_analysis": chart_analysis,
                "youtube_analysis": youtube_analysis,
                "past_reflections": past_reflections  # 반성 일기 데이터 추가
            }


            response = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a cryptocurrency trading analyst. Analyze the provided market data and generate a trading decision."
                    },
                    {
                        "role": "user",
                        "content": f"Market Data Analysis:\n{json.dumps(optimized_data, indent=2)}"
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "trading_decision",
                        "description": "Trading decision based on market analysis",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                 "percentage": {
                                    "type": "integer",
                                    "description": "For buy: Percentage of available KRW to use for purchase. For sell: Percentage of held BTC to sell. For hold: Should be 0. Range: 0-100"
                                },
                               
                                "confidence_score": {
                                    "type": "integer",
                                        "description": "Confidence level of the trading decision (0-100)"
                                    },
                                "decision": {
                                    "type": "string",
                                    "description": "Trading decision to make",
                                    "enum": ["buy", "sell", "hold"]
                                },
                               
                                "reason": {
                                    "type": "string",
                                    "description": "Detailed explanation for the decision"
                                },
                                "reflection_based_adjustments": {
                                "type": "object",
                                    "properties": {
                                        "risk_adjustment": {"type": "string"},
                                        "strategy_improvement": {"type": "string"},
                                        "confidence_factors": {"type": "array", "items": {"type": "string"}}
                                    },
                                    "required": ["risk_adjustment", "strategy_improvement", "confidence_factors"],
                                    "additionalProperties": False
                                }
                            },
                            "required": ["percentage","confidence_score","decision","reason", "reflection_based_adjustments"],
                            "additionalProperties": False
                        }
                    }
                }
            )


            # 응답 파싱
            result = json.loads(response.choices[0].message.content)
           
         
           
            return result


        except Exception as e:
            print(f"Error in get_ai_analysis: {e}")
            return None






    def execute_trade(self, decision, percentage, confidence_score, fear_greed_value,reason):
            """매매 실행 로직"""
            try:
               
                trade_ratio = self.trade_manager.adjust_trade_ratio(percentage, fear_greed_value, decision)


                if confidence_score > 70:
                    if decision == "buy":
                        krw = self.upbit.get_balance("KRW")
                        if krw > 5000:
                            order_amount = krw * trade_ratio
                            order = self.trade_manager.execute_market_buy(order_amount)
                         
                            if order:
                                print("\n=== Buy Order Executed ===")
                                print(f"Trade Amount: {order_amount:,.0f} KRW ({trade_ratio*100:.1f}%)")
                               
                    elif decision == "sell":
                        btc = self.upbit.get_balance(self.ticker)
                        sell_amount = btc * trade_ratio
                        order = self.trade_manager.execute_market_sell(sell_amount)
                       
                        if order:
                            print("\n=== Sell Order Executed ===")
                            print(f"Trade Amount: {sell_amount:.8f} BTC ({trade_ratio*100:.1f}%)")


                # 거래 상태 기록
                balances = self.trade_manager.get_current_balances()
                trade_data = {
                    'decision': decision,
                    'percentage': percentage,
                    'reason': reason,
                    **balances
                }
                self.db.record_trade(trade_data)
               
            except Exception as e:
                print(f"Error in execute_trade: {e}")




def ai_trading():
    try:
        trader = EnhancedCryptoTrader("KRW-BTC")
       
        # 과거 거래 분석 및 반성 수행
        reflection = trader.analyze_past_decisions()
        if reflection:
            print("\n=== Trading Reflection ===")
            print(json.dumps(reflection, indent=2))


       
        current_status = trader.get_current_status()
        orderbook_data = trader.get_orderbook_data()
        ohlcv_data = trader.get_ohlcv_data()
        fear_greed_data = trader.get_fear_greed_index()
        news_data = trader.get_crypto_news()
       
        if all([current_status, orderbook_data, ohlcv_data, fear_greed_data, news_data]):
            analysis_data = {
                "current_status": current_status,
                "orderbook": orderbook_data,
                "ohlcv": ohlcv_data,
                "fear_greed": fear_greed_data,
                "news": news_data
            }
           
            ai_result = trader.get_ai_analysis(analysis_data)
           
            if ai_result:
                print("\n=== AI Analysis Result ===")
                print(json.dumps(ai_result, indent=2))
               
                # 반성 기반 조정사항 출력
                print("\n=== Reflection-based Adjustments ===")
                print(json.dumps(ai_result['reflection_based_adjustments'], indent=2))
               
                trader.execute_trade(
                    ai_result['decision'],
                    ai_result['percentage'],
                    ai_result['confidence_score'],
                    fear_greed_data['current']['value'],
                    ai_result['reason']
                )
       
    except Exception as e:
        print(f"Error in ai_trading: {e}")




# 메인 실행 코드
if __name__ == "__main__":
    try:
        import schedule
        from datetime import datetime
       
        env_type = 'EC2' if is_ec2() else '로컬'
        print(f"Enhanced Bitcoin Trading Bot 시작 ({env_type} 환경)")
        print("종료하려면 Ctrl+C를 누르세요")
       
        load_dotenv()
       
        # 필수 환경 변수 체크
        required_env_vars = ['UPBIT_ACCESS_KEY', 'UPBIT_SECRET_KEY']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"필수 환경 변수가 없습니다: {', '.join(missing_vars)}")


        def run_trading():
            try:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{current_time}] 트레이딩 시작...")
                ai_trading()
                print(f"[{current_time}] 트레이딩 완료")
            except Exception as e:
                print(f"실행 중 오류 발생: {e}")


        # 스케줄 설정
        schedule.every().day.at("09:00").do(run_trading)
        schedule.every().day.at("15:00").do(run_trading)
        schedule.every().day.at("21:00").do(run_trading)


        # 시작 시 즉시 한 번 실행
        print("\n첫 번째 트레이딩 시작...")
        run_trading()
       
        while True:
            try:
                schedule.run_pending()
                time.sleep(30)  # 30초마다 스케줄 체크
            except KeyboardInterrupt:
                print("\n사용자에 의해 봇이 종료되었습니다")
                break
            except Exception as e:
                print(f"실행 중 오류 발생: {e}")
                time.sleep(60)  # 에러 발생시 60초 대기
               
    except Exception as e:
        print(f"프로그램 실행 중 치명적 오류 발생: {e}")