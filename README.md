# 📊 Portfolio Optimization System – Flask

본 리포지토리는 **Flask** 를 기반으로 한 포트폴리오 최적화 시스템의 **백엔드** 부분을 포함하고 있다.  
**재무 데이터**를 활용하여 포트폴리오 최적화를 수행하며,  
**프론트엔드 (React/Next.js)** 와 통신하여 최적화 결과를 제공하는 역할을 한다.

---

## 🚀 주요 기능

- **📡 API 서버**  
  - Flask를 이용하여 **포트폴리오 최적화 API**를 제공한다.  
  - `POST /optimize` 엔드포인트를 통해 최적화 요청을 처리한다.

- **📊 데이터 로딩 및 정제**  
  - `stock_data.csv`, `selected_data.csv`, `cleaned_data.csv` 등의 데이터 파일을 이용하여 **주가 데이터를 처리**한다.  
  - `init_db.py`를 실행하여 초기 데이터를 데이터베이스에 로드할 수 있다.

- **⚡ 최적화 알고리즘 적용**  
  - 다양한 재무 지표(예: **PER, Dividend Yield, Beta, RSI, 거래량, 변동성**)를 고려하여  
    최적의 포트폴리오 구성을 산출한다.

- **🔧 컨테이너화 및 배포 지원**  
  - **Docker & Docker Compose** 를 활용하여 손쉽게 서버를 실행할 수 있다.  
  - **Heroku 또는 AWS 등 클라우드 환경**에 배포 가능하다.

---

## 📁 프로젝트 구조

![image](https://github.com/user-attachments/assets/6cc9343c-1b21-4666-a217-7b901550d4e6)


## 📝 참고  
본 리포지토리는 **백엔드 API 서버** 역할을 수행하며,  
**프론트엔드 (React/Next.js)** 와 통신하여 최적화된 포트폴리오 데이터를 제공한다.

---

## 🛠 설치 및 실행 방법

### 💻 로컬 환경에서 실행

1️⃣ **Python 설치**  
   - Python 3.8 이상이 필요하며, [공식 웹사이트](https://www.python.org/)에서 다운로드 후 설치한다.

2️⃣ **가상 환경 생성 (선택 사항)**  

   python -m venv venv
   source venv/bin/activate  # (Windows: venv\Scripts\activate)

3️⃣ **의존성 패키지 설치**  

- pip install -r requirements.txt

## 4️⃣ 서버 실행

터미널에서 아래 명령어를 실행합니다:

- python app.py

실행 후, [http://localhost:5000](http://localhost:5000) 에서 API를 사용할 수 있습니다.

---

## 🐳 Docker 컨테이너 실행

본 프로젝트는 Docker를 이용한 실행을 지원합니다.

### 1️⃣ Docker 설치
- Docker 공식 웹사이트에서 Docker를 다운로드 및 설치합니다.

### 2️⃣ Docker Compose 실행
터미널에서 아래 명령어를 실행합니다:

- docker-compose up --build


실행 후, [http://localhost:5000](http://localhost:5000) 에서 API를 사용할 수 있습니다.

---

## 🌎 배포 방법

### 🚀 Render를 통한 클라우드 배포

본 프로젝트는 Render를 이용하여 배포되었습니다.  
배포된 서버는 프론트엔드(Vercel)와 통신하여 최적화된 포트폴리오 데이터를 제공합니다.

**배포 단계:**
1. Render 공식 웹사이트에 접속하여 계정을 생성하고 로그인합니다.
2. GitHub 리포지토리를 연결한 후, Python 환경을 선택하여 배포를 진행합니다.
3. `requirements.txt`와 Start Command (`gunicorn app:app`)를 설정합니다.
4. Render에서 제공하는 URL을 통해 API를 사용할 수 있습니다.

---

### 🌐 Vercel을 통한 웹 배포 (프론트엔드)

본 프로젝트의 프론트엔드(React/Next.js) 부분은 Vercel을 통해 배포되었습니다.

**배포 단계:**
1. Vercel 공식 웹사이트에서 계정을 생성하고 로그인합니다.
2. 프론트엔드 리포지토리를 선택하여 Vercel에서 자동 배포를 설정합니다.
3. 배포가 완료되면 제공된 Vercel URL을 통해 웹 애플리케이션을 사용할 수 있습니다.

---

## 📌 기술 스택

| **기술 영역**       | **사용된 기술**                        |
|---------------------|---------------------------------------|
| **Backend**         | Flask (Python), Pandas, NumPy        |
| **Database**        | SQLite (기본), PostgreSQL (배포 시)   |
| **Containerization**| Docker, Docker Compose               |
| **Cloud Deployment**| Render                               |
| **Frontend Deployment** | Vercel                           |

> **참고:** 본 리포지토리는 백엔드 API 서버 역할을 수행하며, 프론트엔드(React/Next.js)와 통신하여 최적화된 포트폴리오 데이터를 제공합니다.


