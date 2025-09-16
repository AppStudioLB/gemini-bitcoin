import os
from dotenv import load_dotenv
import json
import pyupbit
import pandas as pd
import ta
from datetime import datetime
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
import sqlite3
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold


def is_oci():
    """Oracle Cloud Infrastructure VM 환경인지 확인"""
    # OCI VM은 DMI 정보에 고유한 asset tag를 가집니다.
    try:
        with open("/sys/class/dmi/id/chassis_asset_tag", "r") as f:
            asset_tag = f.read().strip()
        return asset_tag == "OracleCloud.com"
    except FileNotFoundError:
        return False
    except Exception:
        return False

def is_ec2():
    """EC2 환경인지 확인 (기존 로직 유지)"""
    try:
        return os.path.exists("/sys/hypervisor/uuid")
    except:
        return False

def is_cloud_vm():
    """주요 클라우드 VM 환경(OCI, EC2)인지 확인"""
    if is_oci():
        return "OCI"
    if is_ec2():
        return "EC2"
    return None

def setup_chrome_options():
    """Chrome 옵션 설정"""
    chrome_options = Options()
   
    # 공통 옵션
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    
    cloud_env = is_cloud_vm()
    if cloud_env:
        # OCI, EC2 등 클라우드 환경에서는 리소스 최적화를 위해 항상 headless 사용
        print(f"클라우드 VM ({cloud_env}) 환경 감지. Headless 모드를 활성화합니다.")
        chrome_options.add_argument("--headless")
    else:
        # 로컬 환경 전용 옵션
        print("로컬 환경 감지. GUI 모드로 시작합니다.")
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
        # 로컬에서도 HEADLESS 환경변수 설정으로 headless 모드 사용 가능
        if os.getenv('HEADLESS', 'false').lower() == 'true':
            print("HEADLESS 환경변수가 설정되어 Headless 모드를 사용합니다.")
            chrome_options.add_argument("--headless")
   
    return chrome_options


def create_driver():
    """WebDriver 생성"""
    try:
        print("ChromeDriver 설정 중...")
        chrome_options = setup_chrome_options()
       
        # EC2 또는 특정 리눅스 환경에서는 chromedriver 경로를 지정할 수 있음
        if is_cloud_vm() and os.path.exists('/usr/bin/chromedriver'):
            service = Service('/usr/bin/chromedriver')
        else:
            try:
                from webdriver_manager.chrome import ChromeDriverManager
                service = Service(ChromeDriverManager().install())
            except ImportError:
                # webdriver-manager가 없는 경우, 시스템 PATH에 의존
                service = Service('chromedriver')
       
        return webdriver.Chrome(service=service, options=chrome_options)
       
    except Exception as e:
        print(f"ChromeDriver 생성 중 오류 발생: {e}")
        raise


class DatabaseManager:
    """데이터베이스 관리를 담당하는 클래스"""
    def __init__(self, db_path="trading.db"):
        self.conn = sqlite3.connect(db_path)
        self.setup_database()
       
    def setup_database(self):
        cursor = self.conn.cursor()
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
        cursor.execute("SELECT * FROM trading_history ORDER BY timestamp DESC LIMIT ?", (limit,))
        return cursor.fetchall()

    def get_reflection_history(self, limit=10):
        """최근 반성 일기 조회"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT r.*, h.decision, h.percentage, h.btc_krw_price
            FROM trading_reflection r JOIN trading_history h ON r.trading_id = h.id
            ORDER BY r.reflection_date DESC LIMIT ?
        """, (limit,))
        return cursor.fetchall()

    def add_reflection(self, reflection_data):
        """반성 일기 추가"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO trading_reflection (trading_id, reflection_date, market_condition,
            decision_analysis, improvement_points, success_rate, learning_points)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            reflection_data['trading_id'], reflection_data['reflection_date'],
            reflection_data['market_condition'], reflection_data['decision_analysis'],
            reflection_data['improvement_points'], reflection_data['success_rate'],
            reflection_data['learning_points']
        ))
        self.conn.commit()

    def record_trade(self, trade_data):
        """거래 데이터를 데이터베이스에 기록"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO trading_history (timestamp, decision, percentage, reason,
            btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(), trade_data['decision'], trade_data['percentage'], trade_data['reason'],
            trade_data['btc_balance'], trade_data['krw_balance'],
            trade_data['btc_avg_buy_price'], trade_data['btc_krw_price']
        ))
        self.conn.commit()
        return cursor.lastrowid

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
            if fear_greed_value <= 25: trade_ratio = min(trade_ratio * 1.2, 1.0)
            elif fear_greed_value >= 75: trade_ratio = trade_ratio * 0.8
        elif trade_type == "sell":
            if fear_greed_value >= 75: trade_ratio = min(trade_ratio * 1.2, 1.0)
            elif fear_greed_value <= 25: trade_ratio = trade_ratio * 0.8
        return trade_ratio

    def get_current_balances(self):
        """현재 잔고 상태 조회"""
        return {
            'btc_balance': float(self.upbit.get_balance(self.ticker)),
            'krw_balance': float(self.upbit.get_balance("KRW")),
            'btc_avg_buy_price': float(self.upbit.get_avg_buy_price(self.ticker)),
            'btc_krw_price': float(pyupbit.get_current_price(self.ticker))
        }

def capture_full_page(url, output_path):
    """웹페이지 캡처 함수"""
    driver = None
    try:
        driver = create_driver()
        wait = WebDriverWait(driver, 20)
        driver.get(url)
        time.sleep(5)
        
        try:
            time_button = wait.until(EC.element_to_be_clickable((By.XPATH, "/html/body/div[1]/div[2]/div[3]/div/section[1]/article[1]/div/span[2]/div/div/div[1]/div[1]/div/cq-menu[1]/span/cq-clickable")))
            time_button.click()
            time.sleep(1)
            hour_option = wait.until(EC.element_to_be_clickable((By.XPATH, "/html/body/div[1]/div[2]/div[3]/div/section[1]/article[1]/div/span[2]/div/div/div[1]/div[1]/div/cq-menu[1]/cq-menu-dropdown/cq-item[8]")))
            hour_option.click()
            time.sleep(3)
        except TimeoutException:
            print("차트 시간 설정을 찾을 수 없습니다. 기본 설정으로 진행합니다.")
        
        total_height = driver.execute_script("return document.body.scrollHeight")
        driver.set_window_size(1920, total_height)
        png = driver.get_screenshot_as_png()
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
    """Gemini 2.5 API를 사용하는 암호화폐 트레이딩 봇"""
    def __init__(self, ticker="KRW-BTC"):
        load_dotenv()
        self.ticker = ticker
        self.access = os.getenv('UPBIT_ACCESS_KEY')
        self.secret = os.getenv('UPBIT_SECRET_KEY')
        self.upbit = pyupbit.Upbit(self.access, self.secret)
        self.trade_manager = TradeManager(self.upbit, ticker)
        self.db = DatabaseManager()
        self.serpapi_key = os.getenv('SERPAPI_KEY')
        self.fear_greed_api = "https://api.alternative.me/fng/"
        self.youtube_channels = ["3XbtEX3jUv4"]

        # Gemini API 설정
        gemini_api_key = os.getenv('GOOGLE_API_KEY')
        if not gemini_api_key:
            raise ValueError("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")
        genai.configure(api_key=gemini_api_key)
        
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        self.gemini_pro_model = genai.GenerativeModel('gemini-2.5-pro')
        
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        self.json_generation_config = GenerationConfig(response_mime_type="application/json")


    def analyze_past_decisions(self):
        """과거 거래 분석 및 반성 (Gemini 사용)"""
        print("analyze_past_decisions")
        try:
            recent_trades = self.db.get_recent_trades(10)
            if not recent_trades:
                print("분석할 과거 거래 내역이 없습니다.")
                return None

            recent_reflections = self.db.get_reflection_history(5)
            current_market = {
                "price": float(pyupbit.get_current_price(self.ticker)),
                "status": self.get_current_status(),
                "fear_greed": self.get_fear_greed_index(),
                "technical": self.get_ohlcv_data()
            }
            
            prompt = f"""You are an AI trading advisor. Analyze these trading records and market conditions. 
            Provide your analysis in JSON format with these exact fields:
            - market_condition: Current market state analysis
            - decision_analysis: Analysis of past trading decisions
            - improvement_points: Points to improve
            - success_rate: numeric value between 0-100
            - learning_points: Key lessons learned

            Data to analyze:
            {json.dumps({"recent_trades": recent_trades, "recent_reflections": recent_reflections, "current_market": current_market}, indent=2, default=str)}
            """
            
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=self.json_generation_config,
                safety_settings=self.safety_settings
            )
            
            reflection = json.loads(response.text)
            
            # 1. improvement_points와 learning_points가 리스트일 경우, 하나의 문자열로 변환
            imp_points = reflection.get('improvement_points', [])
            improvement_points_str = "\n".join(imp_points) if isinstance(imp_points, list) else imp_points

            learn_points = reflection.get('learning_points', [])
            learning_points_str = "\n".join(learn_points) if isinstance(learn_points, list) else learn_points

            # 2. success_rate를 안전하게 숫자로 변환 (이전 해결책 포함)
            raw_rate = reflection.get('success_rate')
            success_rate = 0.0
            try:
                value_to_convert = raw_rate[0] if isinstance(raw_rate, list) and raw_rate else raw_rate
                if value_to_convert is not None:
                    success_rate = float(value_to_convert)
            except (ValueError, TypeError):
                pass
            
            # 3. 최종 데이터 구성
            reflection_data = {
                'trading_id': recent_trades[0][0],
                'reflection_date': datetime.now(),
                'market_condition': reflection.get('market_condition', 'N/A'),
                'decision_analysis': reflection.get('decision_analysis', 'N/A'),
                'improvement_points': improvement_points_str, # 문자열로 변환된 값 사용
                'success_rate': success_rate,
                'learning_points': learning_points_str # 문자열로 변환된 값 사용
            }

            print("\n=== Reflection Data ===")
            print(json.dumps(reflection_data, indent=2, default=str))
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
                data = response.json()['data']
                latest = data[0]
                print(f"\n=== Fear and Greed Index: {latest['value']} ({latest['value_classification']}) ===")
                processed_data = [{'date': datetime.fromtimestamp(int(item['timestamp'])).strftime('%Y-%m-%d'), 'value': int(item['value']), 'classification': item['value_classification']} for item in data]
                values = [d['value'] for d in processed_data]
                avg_value = sum(values) / len(values)
                return {
                    'current': {'value': values[0], 'classification': processed_data[0]['classification']},
                    'history': processed_data,
                    'trend': 'Improving' if values[0] > avg_value else 'Deteriorating',
                    'average': avg_value
                }
            return None
        except Exception as e:
            print(f"Error in get_fear_greed_index: {e}")
            return None

    def add_technical_indicators(self, df):
        """기술적 분석 지표 추가"""
        df['bb_high'] = ta.volatility.BollingerBands(close=df['close']).bollinger_hband()
        df['bb_mid'] = ta.volatility.BollingerBands(close=df['close']).bollinger_mavg()
        df['bb_low'] = ta.volatility.BollingerBands(close=df['close']).bollinger_lband()
        df['bb_pband'] = ta.volatility.BollingerBands(close=df['close']).bollinger_pband()
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close']).rsi()
        df['macd'] = ta.trend.MACD(close=df['close']).macd()
        df['macd_signal'] = ta.trend.MACD(close=df['close']).macd_signal()
        df['macd_diff'] = ta.trend.MACD(close=df['close']).macd_diff()
        df['ma5'] = ta.trend.SMAIndicator(close=df['close'], window=5).sma_indicator()
        df['ma20'] = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
        return df

    def get_current_status(self):
        """현재 투자 상태 조회"""
        try:
            balances = self.trade_manager.get_current_balances()
            total_value = balances['krw_balance'] + (balances['btc_balance'] * balances['btc_krw_price'])
            unrealized_profit = ((balances['btc_krw_price'] - balances['btc_avg_buy_price']) * balances['btc_balance']) if balances['btc_balance'] else 0
            profit_percentage = ((balances['btc_krw_price'] / balances['btc_avg_buy_price']) - 1) * 100 if balances['btc_balance'] and balances['btc_avg_buy_price'] > 0 else 0
            
            print("\n=== Current Investment Status ===")
            print(f"보유 현금: {balances['krw_balance']:,.0f} KRW, 보유 코인: {balances['btc_balance']:.8f} BTC")
            print(f"평균 매수가: {balances['btc_avg_buy_price']:,.0f} KRW, 현재가: {balances['btc_krw_price']:,.0f} KRW")
            print(f"미실현 손익: {unrealized_profit:,.0f} KRW ({profit_percentage:.2f}%)")
            
            return {**balances, "total_value": total_value, "unrealized_profit": unrealized_profit, "profit_percentage": profit_percentage}
        except Exception as e:
            print(f"Error in get_current_status: {e}")
            return None

    def get_orderbook_data(self):
        """호가 데이터 조회"""
        try:
            orderbook_raw = pyupbit.get_orderbook(ticker=self.ticker)
            if not orderbook_raw or 'orderbook_units' not in orderbook_raw:
                return None
            orderbook = orderbook_raw['orderbook_units'][:5]
            return {
                "ask_prices": [u['ask_price'] for u in orderbook], "ask_sizes": [u['ask_size'] for u in orderbook],
                "bid_prices": [u['bid_price'] for u in orderbook], "bid_sizes": [u['bid_size'] for u in orderbook]
            }
        except Exception as e:
            print(f"Error in get_orderbook_data: {e}")
            return None

    def get_ohlcv_data(self):
        """차트 데이터 수집 및 기술적 분석"""
        try:
            daily_data = self.add_technical_indicators(pyupbit.get_ohlcv(self.ticker, interval="day", count=30))
            hourly_data = self.add_technical_indicators(pyupbit.get_ohlcv(self.ticker, interval="minute60", count=24))
            
            print("\n=== Latest Technical Indicators ===")
            print(f"RSI: {daily_data['rsi'].iloc[-1]:.2f}, MACD: {daily_data['macd'].iloc[-1]:.2f}, BB Position: {daily_data['bb_pband'].iloc[-1]:.2f}")
            
            return {
                "daily_data": [dict(r, date=i.strftime('%Y-%m-%d')) for i, r in daily_data.iterrows()][-7:],
                "hourly_data": [dict(r, date=i.strftime('%Y-%m-%d %H:%M:%S')) for i, r in hourly_data.iterrows()][-6:],
                "latest_indicators": {
                    "rsi": daily_data['rsi'].iloc[-1], "macd": daily_data['macd'].iloc[-1],
                    "macd_signal": daily_data['macd_signal'].iloc[-1], "bb_position": daily_data['bb_pband'].iloc[-1]
                }
            }
        except Exception as e:
            print(f"Error in get_ohlcv_data: {e}")
            return None

    def capture_and_analyze_chart(self):
        """차트 캡처 및 분석 (Gemini Vision 사용)"""
        screenshot_path = ""
        try:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"chart_{current_time}.png"
            url = f"https://upbit.com/exchange?code=CRIX.UPBIT.{self.ticker}"
            if not capture_full_page(url, screenshot_path): return None
            
            chart_image = Image.open(screenshot_path)
            prompt = "Analyze this cryptocurrency chart. Provide insights on: 1) Current trend 2) Key support/resistance levels 3) Technical indicator signals 4) Notable patterns"
            
            response = self.gemini_model.generate_content([prompt, chart_image])
            return response.text
        except Exception as e:
            print(f"Error in capture_and_analyze_chart: {e}")
            return None
        finally:
            if screenshot_path and os.path.exists(screenshot_path):
                os.remove(screenshot_path)

    def get_crypto_news(self):
        """비트코인 관련 최신 뉴스 조회"""
        try:
            params = {"engine": "google_news", "q": "bitcoin crypto trading", "api_key": self.serpapi_key, "gl": "us", "hl": "en"}
            response = requests.get("https://serpapi.com/search.json", params=params)
            if response.status_code == 200 and 'news_results' in response.json():
                news = [{'title': n.get('title', ''), 'link': n.get('link', ''), 'source': n.get('source', {}).get('name', ''), 'date': n.get('date', ''), 'snippet': n.get('snippet', '')} for n in response.json()['news_results'][:5]]
                print("\n=== Latest Crypto News ===")
                for n in news: print(f"- {n['title']} ({n['source']})")
                return news
            return None
        except Exception as e:
            print(f"Error in get_crypto_news: {e}")
            return None

    def get_youtube_analysis(self):
        """유튜브 전략 파일 분석 (Gemini 사용)"""
        try:
            with open('strategy.txt', 'r', encoding='utf-8') as f:
                content = f.read()
            
            prompt = """You are an expert cryptocurrency trading analyst. Analyze this Korean content related to cryptocurrency trading.
            Focus on: 1. Trading Strategy (entry/exit, risk management), 2. Market Analysis (sentiment, price levels), 3. Risk Factors, 4. Technical Analysis, 5. Market Impact Factors.
            Provide analysis in JSON format with confidence scores.
            Content to analyze:
            """ + content

            response = self.gemini_model.generate_content(
                prompt,
                generation_config=self.json_generation_config,
                safety_settings=self.safety_settings
            )
            return json.loads(response.text)

        except Exception as e:
            print(f"Error in get_youtube_analysis: {e}")
            return None

    def get_ai_analysis(self, analysis_data):
        """AI 분석 및 매매 신호 생성 (Gemini Structured Outputs 사용)"""
        try:
            optimized_data = {
                "current_status": analysis_data["current_status"], "orderbook": analysis_data["orderbook"],
                "ohlcv": analysis_data["ohlcv"], "fear_greed": analysis_data["fear_greed"],
                "news": analysis_data["news"], "chart_analysis": self.capture_and_analyze_chart(),
                "youtube_analysis": self.get_youtube_analysis(), "past_reflections": self.db.get_reflection_history(5)
            }
            
            trading_decision_schema = {
                "type": "object",
                "properties": {
                    "percentage": {"type": "integer", "description": "Percentage of assets to trade (0-100). 0 for hold."},
                    "confidence_score": {"type": "integer", "description": "Confidence level of the decision (0-100)"},
                    "decision": {"type": "string", "enum": ["buy", "sell", "hold"]},
                    "reason": {"type": "string", "description": "Detailed explanation for the decision"},
                    "reflection_based_adjustments": {
                        "type": "object",
                        "properties": {
                            "risk_adjustment": {"type": "string"},
                            "strategy_improvement": {"type": "string"},
                            "confidence_factors": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["risk_adjustment", "strategy_improvement", "confidence_factors"]
                    }
                },
                "required": ["percentage", "confidence_score", "decision", "reason", "reflection_based_adjustments"]
            }

            prompt = f"""You are a cryptocurrency trading analyst. Analyze the provided market data and generate a trading decision.
            Market Data Analysis:\n{json.dumps(optimized_data, indent=2, default=str)}
            """
            
            response = self.gemini_pro_model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=trading_decision_schema
                ),
                safety_settings=self.safety_settings
            )
            
            return json.loads(response.text)

        except Exception as e:
            print(f"Error in get_ai_analysis: {e}")
            return None

    def execute_trade(self, decision, percentage, confidence_score, fear_greed_value, reason):
        """매매 실행 로직"""
        try:
            trade_ratio = self.trade_manager.adjust_trade_ratio(percentage, fear_greed_value, decision)
            if confidence_score > 70:
                if decision == "buy":
                    order_amount = self.upbit.get_balance("KRW") * trade_ratio
                    if self.trade_manager.execute_market_buy(order_amount):
                        print(f"\n=== Buy Order Executed: {order_amount:,.0f} KRW ({trade_ratio*100:.1f}%) ===")
                elif decision == "sell":
                    sell_amount = self.upbit.get_balance(self.ticker) * trade_ratio
                    if self.trade_manager.execute_market_sell(sell_amount):
                        print(f"\n=== Sell Order Executed: {sell_amount:.8f} BTC ({trade_ratio*100:.1f}%) ===")
            
            trade_data = {'decision': decision, 'percentage': percentage, 'reason': reason, **self.trade_manager.get_current_balances()}
            self.db.record_trade(trade_data)
        except Exception as e:
            print(f"Error in execute_trade: {e}")

def ai_trading():
    try:
        trader = EnhancedCryptoTrader("KRW-BTC")
        reflection = trader.analyze_past_decisions()
        if reflection: print("\n=== Trading Reflection Completed ===")
        
        analysis_data = {
            "current_status": trader.get_current_status(), "orderbook": trader.get_orderbook_data(),
            "ohlcv": trader.get_ohlcv_data(), "fear_greed": trader.get_fear_greed_index(),
            "news": trader.get_crypto_news()
        }
        
        if all(value is not None for value in analysis_data.values()):
            ai_result = trader.get_ai_analysis(analysis_data)
            if ai_result:
                print("\n=== AI Analysis Result ===")
                print(json.dumps(ai_result, indent=2))
                trader.execute_trade(
                    ai_result['decision'], ai_result['percentage'], ai_result['confidence_score'],
                    analysis_data['fear_greed']['current']['value'], ai_result['reason']
                )
        else:
            print("\n데이터 수집 중 일부가 실패하여 이번 트레이딩 주기를 건너뜁니다.")

    except Exception as e:
        print(f"Error in ai_trading: {e}")

if __name__ == "__main__":
    try:
        cloud_env = is_cloud_vm()
        env_type = f"클라우드 ({cloud_env})" if cloud_env else '로컬'
        print(f"Gemini 2.5 Bitcoin Trading Bot 시작 ({env_type} 환경)")
        print("종료하려면 Ctrl+C를 누르세요")
        
        load_dotenv()
        required_vars = ['UPBIT_ACCESS_KEY', 'UPBIT_SECRET_KEY', 'GOOGLE_API_KEY']
        if missing := [v for v in required_vars if not os.getenv(v)]:
            raise ValueError(f"필수 환경 변수가 없습니다: {', '.join(missing)}")

        def run_trading():
            try:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{current_time}] 트레이딩 시작...")
                ai_trading()
                print(f"[{current_time}] 트레이딩 완료")
            except Exception as e:
                print(f"실행 중 오류 발생: {e}")
        
        import schedule
        schedule.every().day.at("09:00").do(run_trading)
        schedule.every().day.at("15:00").do(run_trading)
        schedule.every().day.at("21:00").do(run_trading)

        print("\n첫 번째 트레이딩 시작...")
        run_trading()
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(30)
            except KeyboardInterrupt:
                print("\n사용자에 의해 봇이 종료되었습니다")
                break
            except Exception as e:
                print(f"실행 중 오류 발생: {e}")
                time.sleep(60)
                
    except Exception as e:
        print(f"프로그램 실행 중 치명적 오류 발생: {e}")
