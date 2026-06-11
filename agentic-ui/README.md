# Deepagents UI

Web interface for interacting with the Lucid AI Agent.

## Installation

Install dependencies:
```bash
yarn install
```

## Running the UI

Start the development server
```bash
yarn dev
```

The UI will open at [http://localhost:3000](http://localhost:3000)

## Configuration

The UI is pre-configured to connect to the `scoring_ml` agent. You only need to provide:

**Deployment URL**: The URL where your agent is running (default: `http://127.0.0.1:2024`)

To get this URL, start your agent first:
```bash
cd scoring_ml_agent
langgraph dev
```

The terminal will show:
```
- 🚀 API: http://127.0.0.1:2024
```

Enter this URL in the UI settings dialog when prompted.
