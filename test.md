요청하신 내용을 바탕으로, **로컬 환경에서 기존 커밋 내역을 활용해 봇을 테스트하는 방법**을 깔끔한 마크다운(Markdown) 문서로 정리해 드립니다.

이 문서를 프로젝트의 `README.md`나 `docs/TESTING.md` 등에 추가하여 활용하시면 좋습니다.

---

# 🧪 로컬 리뷰 봇 테스트 가이드 (Mock Request)

GitHub에 매번 코드를 푸시하고 PR을 생성하는 과정 없이, 로컬 환경에서 `curl`을 이용해 가짜 요청(Mock Request)을 보냄으로써 봇의 응답을 즉시 확인할 수 있습니다.

## 📋 전제 조건 (Prerequisites)

1. **리뷰 봇 서버 실행:** 로컬 포트(예: `8000`)에 봇 서버가 실행 중이어야 합니다.
* `python main.py` (Local) 또는 `docker run ...` (Docker)


2. **타겟 레포지토리:** 로컬 경로에 분석할 대상 프로젝트(`escargot`)가 존재해야 합니다.

---

## Scenario A: 기존 커밋 내역으로 테스트 (History Replay)

이미 커밋된 과거의 변경 사항들을 봇이 어떻게 분석하는지 확인하고 싶을 때 사용하는 방법입니다. ("이때 이 코드를 봇이 봤다면 뭐라고 했을까?")

### 1단계: 비교할 커밋 해시(SHA) 확보

`escargot` 프로젝트 폴더에서 로그를 확인하여 비교하고 싶은 두 시점의 해시 값을 복사합니다.

```bash
# escargot 폴더로 이동
cd ~/escargot

# 커밋 로그 확인 (해시값 복사)
git log --oneline

```

* **`base_sha`**: 기준이 되는 과거 시점 (변경 전)
* **`head_sha`**: 변경이 완료된 최신 시점 (변경 후)

### 2단계: Mock Request 전송 (CURL)

터미널에서 아래 명령어를 입력하여 로컬 서버에 리뷰 요청을 보냅니다.

```bash
curl -X POST "http://localhost:8000/review" \
     -H "Content-Type: application/json" \
     -d '{
           "repo_owner": "local_test_user",
           "repo_name": "escargot",
           "pull_request_number": 999,
           "base_sha": "여기에_BASE_SHA_붙여넣기",
           "head_sha": "여기에_HEAD_SHA_붙여넣기"
         }'

```

> **💡 참고:**
> * `repo_owner`, `repo_name`: 로컬 테스트 시에는 실제 경로 매핑만 맞다면 임의의 값을 넣어도 무방합니다.
> * `pull_request_number`: 로컬 테스트 시에는 아무 숫자나 입력해도 됩니다 (로그 식별용).
> 
> 

### 3단계: 결과 확인

서버 로그 또는 `curl`의 응답(JSON)을 통해 봇이 생성한 리뷰 코멘트를 확인합니다.

```json
// 예상 응답 예시
{
  "reviews": [
    {
      "file": "src/parser.cpp",
      "line": 42,
      "comment": "[Memory Optimization] Consider reordering struct members..."
    }
  ]
}

```