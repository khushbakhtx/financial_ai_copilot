FROM langchain/langgraph-api:3.12

COPY pyproject.toml uv.lock /deps/ai_agent/
RUN cd /deps/ai_agent && uv export --no-hashes --no-emit-project > /tmp/requirements.txt \
    && uv pip install --system --no-cache-dir -r /tmp/requirements.txt \
    && uv pip install --system --no-cache-dir "langgraph-api==0.8.5" \
    && uv pip install --system --no-cache-dir "redis==7.4.0"

COPY . /deps/ai_agent/
RUN cd /deps/ai_agent && uv pip install --system --no-cache-dir --no-deps -e .

ENV LANGSERVE_GRAPHS='{"lucid_agent": "/deps/ai_agent/agent.py:agent", "lucid_autonomous": "/deps/ai_agent/agent_auto.py:agent_autonomous"}'
WORKDIR /deps/ai_agent
