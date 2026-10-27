"""
AegisFlow Curated CI/CD Log Dataset
===================================
Provides realistic, multi-scenario CI failure logs across the 4 core triage classes:
1. TRANSIENT_INFRA_FLAKE: Network drops, runner out-of-memory/disk, docker pull rate limits, cloud timeouts.
2. DEPENDENCY_CONFLICT: Incompatible pip wheels, poetry lockfile mismatch, npm peer dependency errors.
3. CODE_REGRESSION: Pytest unit test assertion errors, null pointer dereferences, TypeError, logic bugs.
4. SYNTAX_LINT_ERROR: Linter violations, ruff/flake8 formatting rules, syntax/indentation errors.
"""

from typing import List, Dict, Any
import random

# Category constants
CATEGORY_TRANSIENT_FLAKE = "TRANSIENT_INFRA_FLAKE"
CATEGORY_DEPENDENCY_CONFLICT = "DEPENDENCY_CONFLICT"
CATEGORY_CODE_REGRESSION = "CODE_REGRESSION"
CATEGORY_SYNTAX_LINT = "SYNTAX_LINT_ERROR"

ALL_CATEGORIES = [
    CATEGORY_TRANSIENT_FLAKE,
    CATEGORY_DEPENDENCY_CONFLICT,
    CATEGORY_CODE_REGRESSION,
    CATEGORY_SYNTAX_LINT
]

RAW_LOG_EXAMPLES: List[Dict[str, Any]] = [
    # --------------------------------------------------------------------------
    # 1. TRANSIENT_INFRA_FLAKE Examples (both short & long)
    # --------------------------------------------------------------------------
    {
        "category": CATEGORY_TRANSIENT_FLAKE,
        "exit_code": 143,
        "churn": 5,
        "history": 0.45,
        "log": "Connection timed out after 30000ms\n504 Gateway Time-out\nOperation timed out."
    },
    {
        "category": CATEGORY_TRANSIENT_FLAKE,
        "exit_code": 1,
        "churn": 2,
        "history": 0.40,
        "log": "Error: Failed to establish a new connection: [Errno 110] Connection timed out\n503 Service Unavailable"
    },
    {
        "category": CATEGORY_TRANSIENT_FLAKE,
        "exit_code": 143,
        "churn": 12,
        "history": 0.42,
        "log": """
==> GitHub Actions Runner: Linux x86_64
2026-10-27T18:04:12Z [INFO] Downloading actions/checkout@v4...
2026-10-27T18:05:45Z [ERROR] RequestError: getaddrinfo EAI_AGAIN api.github.com
2026-10-27T18:06:12Z [WARNING] Retrying connection (1 of 3) in 5000ms...
2026-10-27T18:07:00Z [ERROR] Failed to establish a new connection: [Errno 110] Connection timed out
2026-10-27T18:08:15Z ##[error]The operation was canceled.
##[error]The job running on runner Hosted Agent has timed out after 360.0 minutes.
Error: Process completed with exit code 143.
"""
    },
    {
        "category": CATEGORY_TRANSIENT_FLAKE,
        "exit_code": 137,
        "churn": 5,
        "history": 0.35,
        "log": """
Starting Docker container test-runner-isolated...
Container memory limit: 512MB
pytest -v -k "test_large_matrix_multiplication"
tests/unit/test_compute.py::test_large_matrix_multiplication
Killed
/entrypoint.sh: line 44: 12458 Killed                  pytest -v
##[error]Process exited with code 137 (OOMKilled: Out of memory container limit reached).
Runner lost communication with the server. Aborting step.
"""
    },
    {
        "category": CATEGORY_TRANSIENT_FLAKE,
        "exit_code": 1,
        "churn": 8,
        "history": 0.28,
        "log": """
Step 4/12 : RUN apt-get update && apt-get install -y curl
Err:1 http://deb.debian.org/debian bookworm InRelease
  503  Service Unavailable [IP: 199.232.138.132 80]
W: Failed to fetch http://deb.debian.org/debian/dists/bookworm/InRelease  503  Service Unavailable
E: Some index files failed to download. They have been ignored, or old ones used instead.
The command '/bin/sh -c apt-get update && apt-get install -y curl' returned a non-zero code: 100
##[error]Docker pull failed: toomanyrequests: You have reached your pull rate limit.
"""
    },
    {
        "category": CATEGORY_TRANSIENT_FLAKE,
        "exit_code": 1,
        "churn": 2,
        "history": 0.50,
        "log": """
tests/e2e/test_payment_gateway.py::test_stripe_webhook_roundtrip
[urllib3.connectionpool] Retrying (Retry(total=0, connect=None, read=None, redirect=None))
requests.exceptions.ConnectionError: HTTPSConnectionPool(host='sandbox.mock-stripe.internal', port=443):
Max retries exceeded with url: /v1/webhook (Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object>:
Failed to establish a new connection: [Errno 111] Connection refused'))
FAILED tests/e2e/test_payment_gateway.py::test_stripe_webhook_roundtrip - ConnectionRefusedError: Connection refused
"""
    },

    # --------------------------------------------------------------------------
    # 2. DEPENDENCY_CONFLICT Examples
    # --------------------------------------------------------------------------
    {
        "category": CATEGORY_DEPENDENCY_CONFLICT,
        "exit_code": 1,
        "churn": 4,
        "history": 0.10,
        "log": """
pip install -r requirements.txt
Collecting pydantic>=2.10.0 (from -r requirements.txt (line 4))
Collecting langchain-core<0.2.0,>=0.1.0 (from -r requirements.txt (line 12))
INFO: pip is looking at multiple versions of langchain-core to determine which version is compatible with other requirements.
ERROR: Cannot install -r requirements.txt because these package versions have conflicting dependencies.
The conflict is caused by:
    The user requested pydantic>=2.10.0
    langchain-core 0.1.45 depends on pydantic<2.0.0 and >=1.7.4
ERROR: ResolutionImpossible: for -r requirements.txt: pydantic>=2.10.0 and langchain-core==0.1.45
Could not find a version that satisfies the requirement pydantic<2.0.0
"""
    },
    {
        "category": CATEGORY_DEPENDENCY_CONFLICT,
        "exit_code": 1,
        "churn": 1,
        "history": 0.05,
        "log": """
npm ci
npm ERR! code ERESOLVE
npm ERR! ERESOLVE unable to resolve dependency tree
npm ERR!
npm ERR! While resolving: dashboard@2.1.0
npm ERR! Found: react@19.0.0
npm ERR! node_modules/react
npm ERR!   react@"^19.0.0" from the root project
npm ERR!
npm ERR! Could not resolve dependency:
npm ERR! peer react@"^16.8.0 || ^17.0.0 || ^18.0.0" from @visx/scale@3.3.0
npm ERR! Conflicting peer dependency: react@18.3.1
npm ERR! Fix the upstream dependency conflict, or retry with --force or --legacy-peer-deps
"""
    },
    {
        "category": CATEGORY_DEPENDENCY_CONFLICT,
        "exit_code": 1,
        "churn": 6,
        "history": 0.08,
        "log": """
poetry install --no-root
Updating dependencies
Resolving dependencies...
Because my-service depends on torch (^2.5.0) which requires python >=3.9, <3.13,
and my-service depends on Python ^3.13, version solving failed.
SolverProblemError: Because no versions of torch match >=2.5.0
and no matching distribution found for torch in your environment Python 3.13.
Failed building wheel for cryptography
"""
    },
    {
        "category": CATEGORY_DEPENDENCY_CONFLICT,
        "exit_code": 1,
        "churn": 2,
        "history": 0.04,
        "log": "ERROR: Could not find a version that satisfies the requirement torch>=2.5.0 (from versions: none)\nERROR: No matching distribution found for torch>=2.5.0"
    },

    # --------------------------------------------------------------------------
    # 3. CODE_REGRESSION Examples
    # --------------------------------------------------------------------------
    {
        "category": CATEGORY_CODE_REGRESSION,
        "exit_code": 1,
        "churn": 85,
        "history": 0.04,
        "log": """
============================= test session starts ==============================
platform linux -- Python 3.11.8, pytest-8.1.1, pluggy-1.4.0
rootdir: /home/runner/work/backend/backend
collected 14 items

tests/unit/test_billing.py::test_invoice_calculation PASSED              [  7%]
tests/unit/test_billing.py::test_discount_application FAILED             [ 14%]
tests/unit/test_billing.py::test_currency_conversion PASSED              [ 21%]

=================================== FAILURES ===================================
___________________________ test_discount_application __________________________

    def test_discount_application():
        user = User(tier="PRO", credits=100)
        cart = Cart(items=[Item(name="Server Plan", price=50.0)])
>       total = calculate_final_invoice(user, cart)

tests/unit/test_billing.py:42:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
billing/calculator.py:88: in calculate_final_invoice
    discount = user.discount_multiplier * cart.subtotal
E   AttributeError: 'User' object has no attribute 'discount_multiplier'

File "billing/calculator.py", line 88, in calculate_final_invoice
    discount = user.discount_multiplier * cart.subtotal
AttributeError: 'User' object has no attribute 'discount_multiplier'
=========================== short test summary info ============================
FAILED tests/unit/test_billing.py::test_discount_application - AttributeError
========================= 1 failed, 13 passed in 1.45s =========================
"""
    },
    {
        "category": CATEGORY_CODE_REGRESSION,
        "exit_code": 1,
        "churn": 42,
        "history": 0.02,
        "log": """
=================================== FAILURES ===================================
__________________________ test_user_session_expiry ___________________________

session = <Session id='abc-123' expires_at=1730000000>

    def test_user_session_expiry(session):
>       assert session.is_valid(current_timestamp=1730000001) is False
E       AssertionError: assert True is False
E        +  where True = <bound method Session.is_valid of <Session ...>>(1730000001)

tests/auth/test_session.py:65: AssertionError
File "/app/auth/session.py", line 112, in is_valid
    return current_timestamp <= self.expires_at
=========================== short test summary info ============================
FAILED tests/auth/test_session.py::test_user_session_expiry - AssertionError: assert True is False
1 failed, 28 passed in 2.11s
"""
    },
    {
        "category": CATEGORY_CODE_REGRESSION,
        "exit_code": 1,
        "churn": 64,
        "history": 0.06,
        "log": """
=================================== FAILURES ===================================
___________________________ test_tokenize_text_batch ___________________________

    def test_tokenize_text_batch():
        tokenizer = Tokenizer()
>       result = tokenizer.batch_encode(["hello world", None])

tests/ml/test_tokenizer.py:31:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
ml/tokenizer.py:54: in batch_encode
    return [self.encode(text) for text in texts]
ml/tokenizer.py:42: in encode
    words = text.strip().split()
E   TypeError: 'NoneType' object has no attribute 'strip'

File "ml/tokenizer.py", line 42, in encode
    words = text.strip().split()
TypeError: 'NoneType' object has no attribute 'strip'
=========================== short test summary info ============================
FAILED tests/ml/test_tokenizer.py::test_tokenize_text_batch - TypeError: 'NoneType' object has no attribute 'strip'
"""
    },

    # --------------------------------------------------------------------------
    # 4. SYNTAX_LINT_ERROR Examples
    # --------------------------------------------------------------------------
    {
        "category": CATEGORY_SYNTAX_LINT,
        "exit_code": 1,
        "churn": 15,
        "history": 0.12,
        "log": """
ruff check .
api/routes/auth.py:44:17: E999 SyntaxError: invalid syntax. Perhaps you forgot a comma?
    return {"token": token "expires_in": 3600}
                           ^
Found 1 error.
##[error]Process completed with exit code 1.
"""
    },
    {
        "category": CATEGORY_SYNTAX_LINT,
        "exit_code": 1,
        "churn": 8,
        "history": 0.15,
        "log": """
flake8 src/ tests/
src/pipeline/runner.py:102:1: E302 expected 2 blank lines, found 1
src/pipeline/runner.py:145:80: E501 line too long (92 > 79 characters)
src/pipeline/runner.py:180:13: F401 'sys' imported but unused
Found 3 style errors.
##[error]Lint check failed! Please run 'black' and 'ruff' before submitting PR.
"""
    },
    {
        "category": CATEGORY_SYNTAX_LINT,
        "exit_code": 1,
        "churn": 3,
        "history": 0.20,
        "log": """
python -m py_compile aegisflow/agent/orchestrator.py
  File "aegisflow/agent/orchestrator.py", line 28
    def execute_repair(self, incident)
                                      ^
SyntaxError: expected ':'
##[error]Process exited with code 1.
"""
    }
]


def generate_augmented_dataset(multiplier: int = 20) -> List[Dict[str, Any]]:
    """Expands base curated examples into an augmented training dataset."""
    dataset: List[Dict[str, Any]] = []
    rng = random.Random(42)
    for item in RAW_LOG_EXAMPLES:
        for _ in range(multiplier):
            jitter_churn = max(0, item["churn"] + rng.randint(-5, 10))
            jitter_history = max(0.01, min(0.95, item["history"] + rng.uniform(-0.05, 0.05)))

            prefix = ""
            if rng.random() > 0.5:
                prefix = f"2026-10-27T18:0{rng.randint(1, 9)}:22Z [RUNNER] Executing step in workspace /home/runner/work\n"

            dataset.append({
                "category": item["category"],
                "exit_code": item["exit_code"],
                "churn": jitter_churn,
                "history": jitter_history,
                "log": prefix + item["log"]
            })
    rng.shuffle(dataset)
    return dataset
