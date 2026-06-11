"""Financial AI Copilot Tools.

Custom tools for financial data investigation: dataset management, E2B sandbox
execution, MongoDB memory persistence, and domain-specific financial analysis.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolArg, tool
from langgraph.types import Command
from typing_extensions import Annotated

# Inside E2B or Docker the root filesystem is writable; locally it is not.
# _WORKDIR is the base for all agent output (findings, experiments, data).
_WORKDIR = Path(os.getenv("AGENT_WORKDIR", "/tmp/financial_ai"))
_DATA_DIR = _WORKDIR / "data"
_FINDINGS_DIR = _WORKDIR / "findings"
_EXPERIMENTS_DIR = _WORKDIR / "experiments"


# ============================================================
# THINK TOOL (strategic reflection)
# ============================================================

@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Strategic reflection tool for planning and decision-making during investigations.

    Use this after receiving findings to plan next steps, assess completeness,
    or reason through complex financial patterns before taking action.

    Args:
        reflection: Detailed reflection on current investigation status, findings,
                    gaps, and planned next steps.

    Returns:
        Confirmation that reflection was recorded.
    """
    return f"[Reflection recorded at {datetime.now().isoformat()}]\n{reflection}"


# ============================================================
# DATASET TOOLS
# ============================================================

@tool(parse_docstring=True)
def load_dataset_info(dataset_path: str) -> str:
    """Get basic information about a financial dataset without loading it fully.

    Reads the first few rows and returns schema, shape, and column names.
    Use this before delegating to DataProfilingAgent to understand what you're working with.

    Args:
        dataset_path: Path to the CSV or parquet dataset file.

    Returns:
        Dataset schema, shape, column list, and first 3 rows as JSON.
    """
    try:
        import pandas as pd

        if dataset_path.endswith(".parquet"):
            df = pd.read_parquet(dataset_path)
        else:
            df = pd.read_csv(dataset_path, nrows=1000)

        info = {
            "path": dataset_path,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": df.isnull().sum().to_dict(),
            "sample_rows": df.head(3).to_dict(orient="records"),
            "numeric_summary": df.describe().to_dict() if len(df.select_dtypes(include="number").columns) > 0 else {},
        }
        return json.dumps(info, indent=2, default=str)
    except Exception as e:
        return f"Error loading dataset info: {str(e)}"


@tool(parse_docstring=True)
def list_available_datasets() -> str:
    """List all available datasets uploaded by the user.

    Checks MongoDB GridFS first (persistent across sessions), then falls back
    to local /tmp/financial_ai/data/ for any datasets generated this session.

    Returns:
        JSON list of datasets with names, sizes, columns, and local paths.
    """
    try:
        from fin_agent.datasets import list_datasets, download_dataset

        datasets = []

        # MongoDB GridFS datasets (persistent, user-uploaded)
        mongo_datasets = list_datasets()
        for d in mongo_datasets:
            datasets.append({
                "name": d.get("name"),
                "source": "mongodb",
                "gridfs_id": d.get("gridfs_id"),
                "size_mb": d.get("size_mb"),
                "columns": d.get("columns", []),
                "rows_preview": d.get("rows_preview"),
                "uploaded_at": d.get("uploaded_at"),
                "local_path": str(_DATA_DIR / d["name"]) if d.get("name") else None,
            })

        # Also check local data dir for generated datasets this session
        if _DATA_DIR.exists():
            mongo_names = {d["name"] for d in mongo_datasets}
            for fpath in _DATA_DIR.iterdir():
                if fpath.suffix in (".csv", ".parquet", ".json", ".xlsx") and fpath.name not in mongo_names:
                    stat = fpath.stat()
                    datasets.append({
                        "name": fpath.name,
                        "source": "local",
                        "size_mb": round(stat.st_size / 1024 / 1024, 2),
                        "local_path": str(fpath),
                    })

        if not datasets:
            return json.dumps({
                "datasets": [],
                "message": "No datasets found. Upload a dataset via the UI or use generate_sample_financial_dataset().",
            })

        return json.dumps({"datasets": datasets, "count": len(datasets)}, indent=2, default=str)
    except Exception as e:
        return f"Error listing datasets: {str(e)}"


@tool(parse_docstring=True)
def download_dataset(name_or_id: str) -> str:
    """Download a dataset from MongoDB GridFS to local disk so sandbox code can use it.

    Call this before running any run_code that reads a dataset file.
    Returns the local file path to pass into the sandbox code.

    Args:
        name_or_id: Dataset filename (e.g. 'transactions.csv') or its gridfs_id string.

    Returns:
        Local file path string the sandbox can read, or an error message.
    """
    try:
        from fin_agent.datasets import download_dataset as _dl

        local_path = _dl(name_or_id)
        if local_path is None:
            # Try local data dir as fallback
            candidate = _DATA_DIR / name_or_id
            if candidate.exists():
                return str(candidate)
            return f"Dataset '{name_or_id}' not found. Use list_available_datasets() to see available datasets."

        return str(local_path)
    except Exception as e:
        return f"Error downloading dataset: {str(e)}"


@tool(parse_docstring=True)
def delete_dataset(name_or_id: str) -> str:
    """Permanently delete a dataset from MongoDB GridFS and local disk cache.

    Use this when the user asks to remove a dataset to free space, or when a
    dataset is no longer needed. This cannot be undone.

    Args:
        name_or_id: Dataset filename (e.g. 'loans.csv') or its gridfs_id string.

    Returns:
        Confirmation message or error.
    """
    try:
        from fin_agent.datasets import delete_dataset as _del
        deleted = _del(name_or_id)
        if deleted:
            return f"Dataset '{name_or_id}' has been permanently deleted from MongoDB GridFS and local cache."
        return f"Dataset '{name_or_id}' was not found in MongoDB GridFS. Use list_available_datasets() to see what exists."
    except Exception as e:
        return f"Error deleting dataset: {str(e)}"


@tool(parse_docstring=True)
def generate_sample_financial_dataset(dataset_type: str = "fraud", n_rows: int = 10000) -> str:
    """Generate a synthetic financial dataset for testing and demonstration.

    Creates a realistic sample dataset and saves it to /data/ for investigation.

    Args:
        dataset_type: Type of financial dataset to generate ('fraud', 'credit_risk',
            'market', or 'kyc').
        n_rows: Number of rows to generate (default 10000, max 100000).

    Returns:
        Path to the generated dataset and summary statistics.
    """
    import numpy as np
    import pandas as pd

    n_rows = min(n_rows, 100_000)
    rng = np.random.default_rng(42)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    if dataset_type == "fraud":
        n_customers = n_rows // 10
        df = pd.DataFrame({
            "transaction_id": [f"TXN{i:08d}" for i in range(n_rows)],
            "customer_id": rng.integers(0, n_customers, n_rows),
            "merchant_id": rng.integers(0, 500, n_rows),
            "device_id": rng.integers(0, n_rows // 5, n_rows),
            "ip_address": rng.integers(0, n_rows // 3, n_rows),
            "transaction_date": pd.date_range("2023-01-01", periods=n_rows, freq="1min"),
            "amount": np.abs(rng.lognormal(4.0, 1.5, n_rows)).round(2),
            "merchant_category": rng.choice(["retail", "restaurant", "online", "atm", "travel", "gas"], n_rows),
            "country": rng.choice(["US", "UK", "DE", "FR", "CN", "RU", "BR"], n_rows, p=[0.5, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05]),
            "hour_of_day": rng.integers(0, 24, n_rows),
            "is_international": rng.choice([0, 1], n_rows, p=[0.85, 0.15]),
            "card_present": rng.choice([0, 1], n_rows, p=[0.3, 0.7]),
            "customer_age": rng.integers(18, 80, n_rows),
        })
        # Fraud label with realistic correlations (~3% fraud rate)
        fraud_prob = (
            0.02
            + 0.05 * (df["amount"] > 500).astype(float)
            + 0.04 * (df["is_international"] == 1).astype(float)
            + 0.03 * (df["hour_of_day"].isin([0, 1, 2, 3, 4])).astype(float)
            + 0.03 * (df["card_present"] == 0).astype(float)
            + 0.02 * (df["country"].isin(["CN", "RU"])).astype(float)
        )
        df["target"] = (rng.random(n_rows) < fraud_prob.clip(0, 0.8)).astype(int)
        out_path = str(_DATA_DIR / "fraud_transactions.csv")

    elif dataset_type == "credit_risk":
        df = pd.DataFrame({
            "loan_id": [f"LOAN{i:08d}" for i in range(n_rows)],
            "customer_id": rng.integers(0, n_rows // 2, n_rows),
            "loan_amount": np.abs(rng.lognormal(9.5, 1.0, n_rows)).round(0),
            "interest_rate": rng.uniform(3.5, 28.0, n_rows).round(2),
            "loan_term_months": rng.choice([12, 24, 36, 48, 60, 84], n_rows),
            "annual_income": np.abs(rng.lognormal(10.8, 0.8, n_rows)).round(0),
            "debt_to_income": rng.beta(2, 5, n_rows).round(4),
            "credit_score": rng.integers(300, 850, n_rows),
            "employment_years": rng.integers(0, 30, n_rows),
            "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE", "OTHER"], n_rows),
            "loan_purpose": rng.choice(["debt_consolidation", "home_improvement", "business", "medical", "vacation"], n_rows),
            "num_open_accounts": rng.integers(1, 30, n_rows),
            "delinquencies_2yr": rng.integers(0, 5, n_rows),
            "application_date": pd.date_range("2020-01-01", periods=n_rows, freq="2h"),
        })
        default_prob = (
            0.05
            + 0.15 * (df["credit_score"] < 620).astype(float)
            + 0.10 * (df["debt_to_income"] > 0.4).astype(float)
            + 0.08 * (df["delinquencies_2yr"] > 0).astype(float)
            - 0.05 * (df["credit_score"] > 750).astype(float)
        )
        df["target"] = (rng.random(n_rows) < default_prob.clip(0.01, 0.9)).astype(int)
        out_path = str(_DATA_DIR / "credit_risk.csv")

    else:
        return f"Unknown dataset_type '{dataset_type}'. Choose: fraud, credit_risk, market, kyc"

    df.to_csv(out_path, index=False)
    summary = {
        "path": out_path,
        "rows": len(df),
        "columns": list(df.columns),
        "target_rate": round(df["target"].mean(), 4),
        "fraud_count": int(df["target"].sum()),
        "size_mb": round(os.path.getsize(out_path) / 1024 / 1024, 2),
    }
    return json.dumps(summary, indent=2, default=str)


# ============================================================
# INVESTIGATION STATE TOOLS
# ============================================================

@tool(parse_docstring=True)
def save_finding(
    agent_name: str,
    finding_type: str,
    content: str,
    severity: str = "INFO",
) -> str:
    """Persist an important finding to the findings store.

    Use this to record key discoveries during investigation so they can be
    referenced by other agents and included in the final report.

    Args:
        agent_name: Name of the agent recording the finding (e.g., 'DataProfilingAgent').
        finding_type: Category of finding (e.g., 'leakage_risk', 'fraud_pattern',
                      'model_metric', 'data_quality', 'segment_insight').
        content: Detailed description of the finding with supporting evidence.
        severity: Severity level: 'CRITICAL', 'WARNING', 'INFO', 'POSITIVE'.

    Returns:
        Confirmation with finding ID and storage path.
    """
    _FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    finding_id = f"{agent_name.lower().replace(' ', '_')}_{finding_type}_{datetime.now().strftime('%H%M%S')}"
    finding = {
        "id": finding_id,
        "agent": agent_name,
        "type": finding_type,
        "severity": severity,
        "content": content,
        "timestamp": datetime.now().isoformat(),
    }
    path = str(_FINDINGS_DIR / f"{finding_id}.json")
    with open(path, "w") as f:
        json.dump(finding, f, indent=2)

    # Persist to MongoDB with embedding for future semantic search
    try:
        from fin_agent.memory import save_with_embedding, get_mongo_collection
        if get_mongo_collection("findings") is not None:
            save_with_embedding("findings", dict(finding), embed_field="content")
    except Exception:
        pass  # local file save already succeeded; mongo is best-effort

    return f"Finding saved: {finding_id} ({severity}) at {path}"


@tool(parse_docstring=True)
def list_findings(severity_filter: Optional[str] = None) -> str:
    """List all investigation findings recorded so far.

    Args:
        severity_filter: Optional filter by severity level ('CRITICAL', 'WARNING', 'INFO', 'POSITIVE').
                         Pass None to list all findings.

    Returns:
        JSON list of all findings with their metadata.
    """
    findings_dir = str(_FINDINGS_DIR)
    if not os.path.exists(findings_dir):
        return json.dumps({"findings": [], "count": 0})

    findings = []
    for fname in os.listdir(findings_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(findings_dir, fname)
        try:
            with open(fpath) as f:
                finding = json.load(f)
            if severity_filter is None or finding.get("severity") == severity_filter:
                findings.append({
                    "id": finding.get("id"),
                    "agent": finding.get("agent"),
                    "type": finding.get("type"),
                    "severity": finding.get("severity"),
                    "timestamp": finding.get("timestamp"),
                    "preview": finding.get("content", "")[:200],
                })
        except Exception:
            continue

    findings.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return json.dumps({"findings": findings, "count": len(findings)}, indent=2)


@tool(parse_docstring=True)
def save_experiment_result(
    experiment_name: str,
    model_name: str,
    metrics: dict,
    parameters: Optional[dict] = None,
    notes: str = "",
) -> str:
    """Save an ML experiment result for tracking and comparison.

    Args:
        experiment_name: Name of the experiment run (e.g., 'baseline_comparison_v1').
        model_name: Name of the model (e.g., 'XGBoost', 'LightGBM').
        metrics: Dictionary of metric name to value (e.g., {'auc': 0.87, 'ks': 0.52}).
        parameters: Optional model hyperparameters used.
        notes: Optional notes about this experiment.

    Returns:
        Confirmation with experiment ID and path.
    """
    _EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    exp_id = f"{experiment_name}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment = {
        "id": exp_id,
        "experiment_name": experiment_name,
        "model_name": model_name,
        "metrics": metrics,
        "parameters": parameters or {},
        "notes": notes,
        "timestamp": datetime.now().isoformat(),
    }
    path = str(_EXPERIMENTS_DIR / f"{exp_id}.json")
    with open(path, "w") as f:
        json.dump(experiment, f, indent=2)

    # Update the leaderboard
    leaderboard_path = str(_EXPERIMENTS_DIR / "leaderboard.json")
    leaderboard = []
    if os.path.exists(leaderboard_path):
        with open(leaderboard_path) as f:
            leaderboard = json.load(f)
    leaderboard.append({"id": exp_id, "model": model_name, **metrics, "timestamp": experiment["timestamp"]})
    leaderboard.sort(key=lambda x: x.get("auc", 0), reverse=True)
    with open(leaderboard_path, "w") as f:
        json.dump(leaderboard, f, indent=2)

    return f"Experiment saved: {exp_id} at {path}\nLeaderboard updated: {leaderboard_path}"


@tool(parse_docstring=True)
def get_experiment_leaderboard() -> str:
    """Get the current ML experiment leaderboard sorted by AUC.

    Returns:
        JSON leaderboard of all experiments with metrics and rankings.
    """
    leaderboard_path = str(_EXPERIMENTS_DIR / "leaderboard.json")
    if not os.path.exists(leaderboard_path):
        return json.dumps({"leaderboard": [], "message": "No experiments run yet."})

    with open(leaderboard_path) as f:
        leaderboard = json.load(f)

    return json.dumps({"leaderboard": leaderboard, "count": len(leaderboard)}, indent=2)


# ============================================================
# FINANCIAL ANALYSIS TOOLS
# ============================================================

@tool(parse_docstring=True)
def calculate_financial_metrics(
    y_true_path: str,
    y_pred_path: str,
    threshold: float = 0.5,
) -> str:
    """Calculate comprehensive financial model evaluation metrics.

    Computes AUC, KS statistic, Gini coefficient, and business metrics
    at the specified decision threshold.

    Args:
        y_true_path: Path to CSV file with actual labels (column 'y_true').
        y_pred_path: Path to CSV file with predicted probabilities (column 'y_pred').
        threshold: Decision threshold for binary classification (default 0.5).

    Returns:
        JSON with AUC, KS, Gini, F1, precision, recall, and business metrics.
    """
    try:
        import numpy as np
        import pandas as pd
        from sklearn.metrics import (
            auc,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
            roc_curve,
        )

        y_true_df = pd.read_csv(y_true_path)
        y_pred_df = pd.read_csv(y_pred_path)

        y_true = y_true_df["y_true"].values
        y_pred = y_pred_df["y_pred"].values
        y_binary = (y_pred >= threshold).astype(int)

        # Core metrics
        roc_auc = roc_auc_score(y_true, y_pred)
        gini = 2 * roc_auc - 1

        # KS Statistic
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        ks_stat = float(np.max(tpr - fpr))

        # Business metrics at threshold
        precision = precision_score(y_true, y_binary, zero_division=0)
        recall = recall_score(y_true, y_binary, zero_division=0)
        f1 = f1_score(y_true, y_binary, zero_division=0)

        # Capture rate: % of frauds caught at threshold
        fraud_mask = y_true == 1
        capture_rate = float(y_binary[fraud_mask].mean()) if fraud_mask.sum() > 0 else 0.0

        # False positive rate at threshold
        non_fraud_mask = y_true == 0
        fpr_at_thresh = float(y_binary[non_fraud_mask].mean()) if non_fraud_mask.sum() > 0 else 0.0

        metrics = {
            "auc_roc": round(roc_auc, 4),
            "gini": round(gini, 4),
            "ks_statistic": round(ks_stat, 4),
            "f1_score": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "capture_rate": round(capture_rate, 4),
            "false_positive_rate": round(fpr_at_thresh, 4),
            "threshold": threshold,
            "n_samples": len(y_true),
            "fraud_rate": round(float(y_true.mean()), 4),
        }
        return json.dumps(metrics, indent=2)
    except Exception as e:
        return f"Error calculating metrics: {str(e)}"


@tool(parse_docstring=True)
def compute_population_stability_index(
    baseline_path: str,
    current_path: str,
    score_col: str = "score",
    n_buckets: int = 10,
) -> str:
    """Compute Population Stability Index (PSI) between baseline and current score distributions.

    PSI < 0.1: Stable (no change)
    PSI 0.1-0.25: Slight shift (monitoring needed)
    PSI > 0.25: Significant shift (model refresh needed)

    Args:
        baseline_path: CSV path with baseline score distribution.
        current_path: CSV path with current score distribution.
        score_col: Column name containing the scores.
        n_buckets: Number of buckets for PSI calculation (default 10).

    Returns:
        JSON with PSI value, bucket analysis, and stability verdict.
    """
    try:
        import numpy as np
        import pandas as pd

        baseline = pd.read_csv(baseline_path)[score_col].dropna().values
        current = pd.read_csv(current_path)[score_col].dropna().values

        breakpoints = np.percentile(baseline, np.linspace(0, 100, n_buckets + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        base_counts = np.histogram(baseline, bins=breakpoints)[0].astype(float) + 1e-10
        curr_counts = np.histogram(current, bins=breakpoints)[0].astype(float) + 1e-10

        base_pct = base_counts / base_counts.sum()
        curr_pct = curr_counts / curr_counts.sum()

        psi_buckets = (curr_pct - base_pct) * np.log(curr_pct / base_pct)
        psi = float(psi_buckets.sum())

        verdict = "STABLE" if psi < 0.1 else "SLIGHT_SHIFT" if psi < 0.25 else "SIGNIFICANT_SHIFT"
        result = {
            "psi": round(psi, 4),
            "verdict": verdict,
            "n_buckets": n_buckets,
            "baseline_n": len(baseline),
            "current_n": len(current),
            "bucket_contributions": [round(float(x), 4) for x in psi_buckets],
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error computing PSI: {str(e)}"


# ============================================================
# CODE EXECUTION TOOL
# ============================================================

@tool(parse_docstring=True)
def run_code(
    code: str,
    timeout: int = 300,
) -> str:
    """Execute Python code in a secure sandbox for financial analysis.

    Runs arbitrary Python code in an E2B sandbox (or local subprocess fallback)
    with pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, optuna,
    shap, matplotlib, seaborn, networkx, and pyod pre-available.

    Use this to run data profiling, feature engineering, ML training, anomaly
    detection, SHAP analysis, or any computational workload. Print results to
    stdout — they will be returned. Save outputs to /tmp/financial_ai/.

    Args:
        code: Valid Python code to execute. Must be self-contained — imports,
            data loading, and computation all in one block.
        timeout: Execution timeout in seconds (default 300, max 600).

    Returns:
        Stdout output from the code execution, plus stderr and artifact list if any.
    """
    from fin_agent.sandbox import run_financial_analysis
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import ToolException

    timeout = min(timeout, 600)
    # Extract agent name from LangChain run metadata if available
    _cfg: RunnableConfig = {}  # populated by LangChain at call time via InjectedToolArg
    agent_label = ""
    try:
        import inspect
        frame = inspect.currentframe()
        # Walk up frames looking for a LangChain config dict
        for _ in range(10):
            if frame is None:
                break
            locs = frame.f_locals
            if "config" in locs and isinstance(locs["config"], dict):
                cfg = locs["config"]
                agent_label = (
                    cfg.get("metadata", {}).get("agent_name", "")
                    or cfg.get("run_name", "")
                )
                if agent_label:
                    break
            frame = frame.f_back
    except Exception:
        pass

    result = run_financial_analysis(code, timeout=timeout, agent=agent_label)

    parts = []
    if result.get("stdout"):
        parts.append(result["stdout"])
    if result.get("stderr"):
        parts.append(f"[stderr]\n{result['stderr']}")
    if result.get("artifacts"):
        parts.append(f"[artifacts] {result['artifacts']}")
    if not result.get("success"):
        parts.append(f"[exit_code={result.get('exit_code', '?')}]")

    return "\n".join(parts) if parts else "(no output)"


# ============================================================
# REPORT STATE TOOL
# ============================================================

@tool(parse_docstring=True)
def save_report(
    file_path: str,
    display_name: Optional[str] = None,
    tool_call_id: Annotated[str, InjectedToolArg] = "",
) -> Union[Command, str]:
    """Read a report file from disk and publish it to the UI files panel.

    Call this after writing any final report or summary file so the user can
    view the content directly in the chat interface. Supports .md, .txt, .json,
    .html, .csv files.

    Args:
        file_path: Absolute path to the file to publish (e.g. '/tmp/final_report.md').
        display_name: Optional short name shown in the UI (e.g. 'final_report.md').
                      Defaults to the filename portion of file_path.

    Returns:
        Confirmation that the file was published to the UI.
    """
    path = Path(file_path)
    if not path.exists():
        return f"File not found: {file_path}"

    name = display_name or path.name
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"

    # Truncate very large files to avoid bloating state (keep first 200k chars)
    if len(content) > 200_000:
        content = content[:200_000] + "\n\n[... truncated — file too large ...]"

    return Command(update={
        "files": {name: content},
        "messages": [ToolMessage(content=f"Report published: {name}", tool_call_id=tool_call_id)],
    })


# ============================================================
# COPILOTKIT STATE UPDATE TOOLS
# These push live updates into the shared state that the frontend reads.
# ============================================================

@tool(parse_docstring=True)
def update_pipeline_step_status(
    step: str,
    status: str,
    summary: str = "",
    tool_call_id: Annotated[str, InjectedToolArg] = "",
) -> Command:
    """Update the UI status of a pipeline step in real-time.

    Call this immediately before starting a step (status='running') and
    immediately after it completes (status='completed' or 'error').
    The frontend renders a live pipeline timeline from this state.

    Args:
        step: Step identifier e.g. '01_eda', '04_baseline_model'.
        status: One of 'pending', 'running', 'completed', 'error', 'skipped'.
        summary: Short one-sentence result summary shown in the timeline.

    Returns:
        Command to update pipeline_steps in shared state.
    """
    now = datetime.now().isoformat()
    return Command(update={
        "_pipeline_step_update": {"step": step, "status": status, "summary": summary, "ts": now},
        "messages": [ToolMessage(content=f"Pipeline step {step} → {status}", tool_call_id=tool_call_id)],
    })


@tool(parse_docstring=True)
def push_model_leaderboard(
    leaderboard_json: str,
    tool_call_id: Annotated[str, InjectedToolArg] = "",
) -> Command:
    """Push the model leaderboard into shared state so the UI can render it.

    Call this after step 04 (baseline models) and after step 06 (tuned model).
    The frontend renders an interactive ModelLeaderboard component from this data.

    Args:
        leaderboard_json: JSON array of model results. Each entry must have:
            model_name (str), auc (float), and optionally gini, ks, f1 (floats).

    Returns:
        Command to update model_leaderboard in shared state.
    """
    try:
        results = json.loads(leaderboard_json)
        ranked = sorted(results, key=lambda x: x.get("auc", 0), reverse=True)
        for i, r in enumerate(ranked):
            r["rank"] = i + 1
        best_auc = ranked[0].get("auc") if ranked else None
        return Command(update={
            "model_leaderboard": ranked,
            "best_auc": best_auc,
            "messages": [ToolMessage(content=f"Leaderboard updated. Best AUC: {best_auc}", tool_call_id=tool_call_id)],
        })
    except Exception as e:
        return Command(update={
            "model_leaderboard": [],
            "messages": [ToolMessage(content=f"Leaderboard update failed: {e}", tool_call_id=tool_call_id)],
        })


@tool(parse_docstring=True)
def set_investigation_context(
    investigation_id: str,
    dataset_name: str,
    pipeline_type: str,
    pipeline_steps_json: str,
    dataset_shape_json: str = "",
    tool_call_id: Annotated[str, InjectedToolArg] = "",
) -> Command:
    """Initialize the shared investigation context at the start of a pipeline run.

    Call this once after the orchestrator has chosen the pipeline and before
    dispatching step 01. This populates the frontend header and pipeline timeline.

    Args:
        investigation_id: Unique ID for this run e.g. 'loans_20260529_143022'.
        dataset_name: Dataset filename e.g. 'credit_risk_10000.csv'.
        pipeline_type: 'credit_scoring', 'fraud_detection', or 'general'.
        pipeline_steps_json: JSON array of PipelineStep objects defining all steps.
        dataset_shape_json: Optional JSON [rows, cols] from load_dataset_info.

    Returns:
        Command to update investigation context fields in shared state.
    """
    update: dict = {
        "investigation_id": investigation_id,
        "dataset_name": dataset_name,
        "pipeline_type": pipeline_type,
    }
    if pipeline_steps_json:
        try:
            update["pipeline_steps"] = json.loads(pipeline_steps_json)
        except Exception:
            pass
    if dataset_shape_json:
        try:
            update["dataset_shape"] = json.loads(dataset_shape_json)
        except Exception:
            pass
    update["messages"] = [ToolMessage(
        content=f"Investigation context set: {investigation_id} ({pipeline_type})",
        tool_call_id=tool_call_id,
    )]
    return Command(update=update)


@tool(parse_docstring=True)
def store_charts(
    charts_json: str,
    title: str = "",
    tool_call_id: Annotated[str, InjectedToolArg] = "",
) -> Command:
    """Store dataset chart specs in shared state so the UI renders them as interactive Plotly charts.

    Call this after generating charts in run_code. Pass the charts as a JSON string.
    The frontend automatically renders the charts when this state field is set.

    Args:
        charts_json: JSON array string of chart specs with keys chart_title, data, layout. Use fig.to_dict() in run_code to generate.
        title: Optional section heading shown above the charts e.g. 'Credit Risk EDA'.

    Returns:
        Confirmation that charts were stored for rendering.
    """
    return Command(update={
        "charts_json": charts_json,
        "charts_title": title,
        "messages": [ToolMessage(content=f"Charts stored for rendering: {title}", tool_call_id=tool_call_id)],
    })


# ============================================================
# SCRIPT EXECUTION TOOL
# ============================================================

@tool(parse_docstring=True)
def run_script(
    script_name: str,
    dataset_path: str = "",
    target_col: str = "",
    investigation_id: str = "",
    extra_args: str = "",
    timeout: int = 300,
) -> str:
    """Execute a named pipeline script from the credit-scoring or fraud-detection skill.

    Scripts live in .deepagents/skills/<pipeline>/. Each script reads its inputs
    from MongoDB pipeline_state (prior step outputs) and writes its results back
    to MongoDB so the next step can pick up where this one left off.

    Args:
        script_name: Script to run, e.g. 'credit-scoring-pipeline/01_eda' or
            'fraud-detection-pipeline/03_graph_analysis'. No .py extension needed.
        dataset_path: Absolute local path to the dataset file (e.g. '/tmp/financial_ai/data/loans.csv').
        target_col: Name of the target/label column in the dataset.
        investigation_id: Unique ID for this investigation run (used as MongoDB partition key).
            If empty, one is generated from dataset name + timestamp.
        extra_args: Any additional key=value pairs to pass as script variables,
            space-separated (e.g. 'test_size=0.2 n_estimators=300').
        timeout: Execution timeout in seconds (default 300, max 600).

    Returns:
        Script stdout/stderr and a summary of what was written to MongoDB.
    """
    import inspect
    from pathlib import Path
    from fin_agent.sandbox import run_financial_analysis

    timeout = min(timeout, 600)

    # Resolve agent dir — skills live relative to agent/
    agent_dir = Path(__file__).parent.parent
    script_path_no_ext = agent_dir / ".deepagents" / "skills" / script_name
    script_path = script_path_no_ext.with_suffix(".py")
    if not script_path.exists():
        script_path = Path(str(script_path_no_ext) + ".py")
    if not script_path.exists():
        return f"Script not found: {script_name}\nLooked at: {script_path}"

    # Auto-generate investigation_id if not provided
    if not investigation_id:
        from datetime import datetime
        ds_stem = Path(dataset_path).stem if dataset_path else "investigation"
        investigation_id = f"{ds_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Build the injected header that every script can read via os.environ
    header = f"""
import os, sys
os.environ.setdefault('DATASET_PATH', {repr(dataset_path)})
os.environ.setdefault('TARGET_COL', {repr(target_col)})
os.environ.setdefault('INVESTIGATION_ID', {repr(investigation_id)})
os.environ.setdefault('MONGODB_URI', os.getenv('MONGODB_URI', ''))
os.environ.setdefault('GEMINI_API_KEY', os.getenv('GEMINI_API_KEY', ''))
"""
    # Parse extra_args (key=value pairs)
    for pair in extra_args.split():
        if "=" in pair:
            k, v = pair.split("=", 1)
            header += f"os.environ.setdefault({repr(k.upper())}, {repr(v)})\n"

    script_body = script_path.read_text(encoding="utf-8")
    full_code = header + "\n" + script_body

    # Extract agent name from LangChain context
    agent_label = ""
    try:
        frame = inspect.currentframe()
        for _ in range(10):
            if frame is None:
                break
            locs = frame.f_locals
            if "config" in locs and isinstance(locs["config"], dict):
                cfg = locs["config"]
                agent_label = cfg.get("metadata", {}).get("agent_name", "") or cfg.get("run_name", "")
                if agent_label:
                    break
            frame = frame.f_back
    except Exception:
        pass

    result = run_financial_analysis(full_code, timeout=timeout, agent=agent_label)

    parts = [f"[script: {script_name}] [investigation: {investigation_id}]"]
    if result.get("stdout"):
        parts.append(result["stdout"])
    if result.get("stderr"):
        parts.append(f"[stderr]\n{result['stderr']}")
    if result.get("artifacts"):
        parts.append(f"[artifacts] {result['artifacts']}")
    if not result.get("success"):
        parts.append(f"[exit_code={result.get('exit_code', '?')}]")

    return "\n".join(parts) if parts else "(no output)"


# ============================================================
# PIPELINE STATE TOOLS (MongoDB-backed step store)
# ============================================================

@tool(parse_docstring=True)
def save_pipeline_step(
    investigation_id: str,
    step: str,
    data: dict,
    dataset_name: str = "",
    pipeline: str = "credit-scoring-pipeline",
) -> str:
    """Persist the output of a pipeline step to MongoDB.

    Every pipeline script calls this at the end of its run. The next step
    reads it back with get_pipeline_state(). This makes the pipeline resumable —
    if a step fails, re-running it overwrites the stale entry cleanly.

    Args:
        investigation_id: Unique ID for this investigation run (partition key).
        step: Step identifier, e.g. '01_eda', '04_baseline_model'.
        data: Dictionary of results to persist (metrics, feature lists, config, etc.).
        dataset_name: Original dataset filename (for display and lookup).
        pipeline: Pipeline name, e.g. 'credit-scoring-pipeline'.

    Returns:
        Confirmation with MongoDB document ID or error message.
    """
    from datetime import datetime

    doc = {
        "investigation_id": investigation_id,
        "pipeline": pipeline,
        "step": step,
        "dataset_name": dataset_name,
        "data": data,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        from fin_agent.memory import get_mongo_collection
        col = get_mongo_collection("pipeline_state")
        if col is None:
            return f"[pipeline_state] MongoDB not configured — step '{step}' not persisted."

        # Upsert: replace any previous result for same investigation+step
        col.replace_one(
            {"investigation_id": investigation_id, "step": step},
            doc,
            upsert=True,
        )
        return f"[pipeline_state] Step '{step}' saved for investigation '{investigation_id}'."
    except Exception as e:
        return f"[pipeline_state] Save error: {e}"


@tool(parse_docstring=True)
def get_pipeline_state(
    investigation_id: str,
    step: Optional[str] = None,
) -> str:
    """Retrieve pipeline state from MongoDB for a given investigation.

    Use this at the start of each pipeline step to load prior step outputs,
    or to check which steps have already been completed before re-running.

    Args:
        investigation_id: Unique ID of the investigation to look up.
        step: Optional specific step to fetch (e.g. '01_eda'). If None,
            returns all completed steps for this investigation.

    Returns:
        JSON with step data, or a list of all completed steps with timestamps.
    """
    try:
        from fin_agent.memory import get_mongo_collection
        col = get_mongo_collection("pipeline_state")
        if col is None:
            return json.dumps({"error": "MongoDB not configured", "steps": []})

        query = {"investigation_id": investigation_id}
        if step:
            query["step"] = step

        docs = list(col.find(query, {"_id": 0}).sort("timestamp", 1))
        if not docs:
            msg = f"No pipeline state found for investigation '{investigation_id}'"
            if step:
                msg += f" step '{step}'"
            return json.dumps({"message": msg, "steps": []})

        return json.dumps({"investigation_id": investigation_id, "steps": docs, "count": len(docs)}, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "steps": []})


# ============================================================
# MONGODB MEMORY TOOLS
# ============================================================

@tool(parse_docstring=True)
def save_to_memory(
    collection: str,
    document: dict,
    investigation_id: Optional[str] = None,
) -> str:
    """Save a document to MongoDB Atlas for persistent organizational memory.

    Use this to persist findings, experiment results, agent reasoning traces,
    and reports so they are available across future investigations.

    Args:
        collection: MongoDB collection name. One of:
            'findings', 'experiments', 'reports', 'reasoning_traces',
            'agent_memory', 'fraud_relationships', 'model_registry'.
        document: Dictionary containing the data to persist.
        investigation_id: Optional ID to link this document to a specific investigation.

    Returns:
        Confirmation with the inserted document ID or error message.
    """
    try:
        from fin_agent.memory import save_with_embedding, get_mongo_collection

        col = get_mongo_collection(collection)
        if col is None:
            return f"MongoDB not configured. Document would be saved to collection '{collection}'. Set MONGODB_URI in .env to enable persistence."

        doc = {**document, "timestamp": datetime.now().isoformat()}
        if investigation_id:
            doc["investigation_id"] = investigation_id

        # Embed the most meaningful text field for vector search
        embed_field = next(
            (f for f in ("content", "summary", "description", "text") if f in doc),
            None,
        )
        if embed_field:
            inserted_id = save_with_embedding(collection, doc, embed_field=embed_field)
        else:
            result = col.insert_one(doc)
            inserted_id = str(result.inserted_id)

        return f"Saved to MongoDB collection '{collection}': _id={inserted_id}"
    except Exception as e:
        return f"MongoDB save error: {str(e)}"


@tool(parse_docstring=True)
def search_memory(
    query: str,
    collection: str = "findings",
    limit: int = 5,
) -> str:
    """Search MongoDB memory for similar past investigations and findings.

    Uses vector similarity search when available, falls back to text search.

    Args:
        query: Natural language search query (e.g., 'credit card fraud patterns').
        collection: MongoDB collection to search ('findings', 'reports', 'reasoning_traces').
        limit: Maximum number of results to return (default 5).

    Returns:
        JSON list of similar past findings or investigations.
    """
    try:
        from fin_agent.memory import vector_search

        results = vector_search(query=query, collection=collection, limit=limit)
        if results is None:
            return json.dumps({"results": [], "message": "MongoDB not configured. Set MONGODB_URI in .env to enable memory search."})
        return json.dumps({"results": results, "count": len(results)}, indent=2, default=str)
    except Exception as e:
        return f"Memory search error: {str(e)}"
