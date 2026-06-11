FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock /app/

# Install all dependencies declared in the root pyproject.toml
RUN uv pip install --system --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "httpx>=0.27.0" \
    "python-dotenv>=1.0.0" \
    "asyncpg>=0.29.0" \
    "langgraph-checkpoint-postgres>=2.0.0" \
    "psycopg[binary,pool]>=3.1.1" \
    "langgraph>=0.3.0" \
    "langchain>=0.3.0" \
    "langchain-core>=0.3.0" \
    "langchain-openai>=1.0.2" \
    "langchain-google-genai>=2.0.0" \
    "langchain-anthropic>=0.3.0" \
    "deepagents>=0.4.5" \
    "pydantic>=2.0.0" \
    "tenacity>=8.0.0" \
    "requests>=2.31.0" \
    "pandas>=2.0.0" \
    "scipy>=1.11.0" \
    "scikit-learn>=1.4.0" \
    "pypdf>=3.17.0" \
    "beautifulsoup4>=4.12.0" \
    "pymupdf>=1.26.7" \
    "e2b-code-interpreter>=2.4.1" \
    "great-expectations>=1.0.0" \
    "websockets>=12.0" \

# Copy the full ai_agent directory:
#   agent.py, agent_auto.py, langgraph.json, ai_agent/, backend/, pyproject.toml, ...
# graphs.py resolves langgraph.json one directory above backend/, so the layout must be:
#   /app/langgraph.json
#   /app/agent.py
#   /app/backend/main.py  ← WORKDIR
COPY . /app/

# Install the ai_agent package so `from ai_agent.tools import ...` resolves
RUN uv pip install --system --no-cache-dir --no-deps -e /app

# backend/ is the working directory; graphs.py walks up one level to find langgraph.json
WORKDIR /app/backend

EXPOSE 8123

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8123"]
