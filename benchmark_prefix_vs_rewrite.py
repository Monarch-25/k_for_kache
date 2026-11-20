#!/usr/bin/env python3
"""
Prefix-Cache Invalidation Benchmark for Local Agent Loops
=========================================================

Hypothesis
----------
On a runner that reuses exact prefixes (vLLM Automatic Prefix Caching), the
marginal cost of *reading new text* is much smaller than the cost of
*invalidating text the runner already processed*:

  Arm A  append_only            : identical prefix + new tool result
  Arm B  rewrite_equal_tokens   : early summary rewrite + same tool result,
                                  total prompt tokens equal to A (+/-2)
  Arm C  rewrite_shorter_prompt : compressed early summary + same tool result,
                                  total prompt ~30% shorter than A

Measured per Turn-2 request:
  * Time To First Token (TTFT, perf_counter, streaming)
  * End-to-end latency and inter-token latency
  * Input tokens, cached tokens, uncached (newly prefetched) tokens
    - from OpenAI-compatible `usage.prompt_tokens_details.cached_tokens`
    - from vLLM Prometheus `vllm:prefix_cache_hits` / `..._queries` deltas
  * Deterministic tool-evidence answer correctness (updated fact used,
    stale decoy fact not cited)

Design notes
------------
* Each (scenario, repetition, arm) gets a unique session salt in the system
  prompt. Priming therefore never leaks KV blocks into another trial.
* For every measured request, the original Turn-1 conversation is primed
  immediately beforehand. Arms B/C then rewrite the early summary, which
  invalidates the downstream block-hash chain -- exactly the behaviour of a
  runner that has already processed the old prefix.
* Prompts are built and verified with the exact Hugging Face tokenizer.
  Arm B total prompt tokens are tuned to Arm A within +/-2 tokens using
  neutral boundary padding; Arm C is tuned to ~70% of Arm A.
* Concurrency = 1, temperature = 0, max_tokens = 512 (config-driven).

Usage
-----
  # full run (starts vLLM if needed, runs benchmark, writes results/)
  python benchmark_prefix_vs_rewrite.py --config run.yaml

  # analyze existing raw results only (no server required)
  python benchmark_prefix_vs_rewrite.py --config run.yaml --analyze-only

  # prompt-construction sanity check (tokenizer required, no server needed)
  python benchmark_prefix_vs_rewrite.py --config run.yaml --dry-run
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.metadata
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
import numpy as np
import yaml

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None  # type: ignore

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # type: ignore


# =============================================================================
# Constants
# =============================================================================

ARM_APPEND = "append_only"
ARM_EQUAL = "rewrite_equal_tokens"
ARM_SHORTER = "rewrite_shorter_prompt"
ARMS = [ARM_APPEND, ARM_EQUAL, ARM_SHORTER]
ARM_CODE = {ARM_APPEND: "a", ARM_EQUAL: "e", ARM_SHORTER: "s"}
ARM_LABEL = {
    ARM_APPEND: "A · append_only",
    ARM_EQUAL: "B · rewrite_equal_tokens",
    ARM_SHORTER: "C · rewrite_shorter_prompt",
}
ARM_COLOR = {
    ARM_APPEND: "#2A6FDB",
    ARM_EQUAL: "#D64545",
    ARM_SHORTER: "#D89B00",
}

# Raw-record field -> friendly metric name used in summary tables.
METRIC_FIELDS = {
    "total_input_tokens": "prompt_tokens",
    "cached_input_tokens_usage": "cached_tokens_usage",
    "cached_input_tokens_metrics": "cached_tokens_metrics",
    "uncached_input_tokens_usage": "uncached_tokens_usage",
    "uncached_input_tokens_metrics": "uncached_tokens_metrics",
    "ttft_ms": "ttft_ms",
    "e2e_ms": "e2e_ms",
    "itl_ms": "itl_ms",
    "completion_tokens": "completion_tokens",
}


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG: Dict[str, Any] = {
    "vllm_server": {
        "host": "127.0.0.1",
        "port": 8000,
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "served_model_name": "Qwen2.5-14B-Instruct",
        "dtype": "bfloat16",
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.90,
        "enable_prefix_caching": True,
        "enable_prompt_tokens_details": True,
        "block_size": 16,
        "trust_remote_code": False,
        "startup_timeout_s": 3600,
        "request_timeout_s": 900,
        "extra_args": [],
    },
    "benchmark": {
        "seed": 42,
        "warmup_trials": 5,
        "num_scenarios": 10,
        "trials_per_scenario": 5,
        "concurrency": 1,
        "temperature": 0.0,
        "max_tokens": 128,
        "prime_max_tokens": 8,
        "prefix_target_tokens": 5500,
        "tool_output_target_tokens": 2200,
        "token_parity_tolerance": 2,
        "shorter_prompt_target_ratio": 0.70,
        "shorter_prompt_min_ratio": 0.65,
        "shorter_prompt_max_ratio": 0.75,
        "request_retries": 2,
        "prime_settle_s": 0.05,
        "metrics_settle_s": 0.20,
    },
    "outputs": {
        "dir": "results",
        "raw_jsonl": "results/raw_trials.jsonl",
        "summary_csv": "results/summary_metrics.csv",
        "summary_json": "results/summary_metrics.json",
        "server_log": "results/vllm_server.log",
        "serve_config_json": "results/serve_config.json",
        "telemetry_json": "results/server_telemetry.json",
        "charts": {
            "ttft_comparison": "results/ttft_comparison.png",
            "uncached_tokens_vs_ttft": "results/uncached_tokens_vs_ttft.png",
            "shorter_prompt_paradox": "results/shorter_prompt_paradox.png",
        },
        "report_md": "REPORT.md",
        "notebook": "prefill_cache_benchmark.ipynb",
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str) -> Tuple[Dict[str, Any], Path]:
    config_path = Path(path).resolve()
    with open(config_path, "r", encoding="utf-8") as fh:
        user_cfg = yaml.safe_load(fh) or {}
    cfg = deep_merge(DEFAULT_CONFIG, user_cfg)
    cfg["_config_path"] = str(config_path)
    cfg["_root"] = str(config_path.parent)
    return cfg, config_path.parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(path: str, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def ensure_parent(path: str, root: Path) -> Path:
    p = resolve_path(path, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except (TypeError, ValueError):
        pass
    return None


def pct(values: Sequence[float], q: float) -> Optional[float]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return None
    return float(np.percentile(arr, q))


def stats_dict(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": None, "median": None, "p95": None, "std": None, "min": None, "max": None}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 5000, seed: int = 42) -> List[Optional[float]]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return [None, None]
    if arr.size == 1:
        return [float(arr[0]), float(arr[0])]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]

# =============================================================================
# Scenario definitions: 10 realistic multi-turn engineering/ops trajectories
# =============================================================================

SCENARIO_SPECS: List[Dict[str, Any]] = [
    {
        "sid": "s01_pytest_timeout",
        "domain": "CI test harness",
        "component": "integration test runner",
        "repo": "acme/checkout-service",
        "service": "checkout-api",
        "symbol": "retry_dispatch",
        "paths": ["tests/integration/test_checkout.py", "tests/conftest.py", "src/checkout/pipeline.py", "pytest.ini"],
        "commands": ["pytest -q tests/integration/test_checkout.py -x", "pytest -q -x -k retry_soft_decline", "pytest --collect-only -q"],
        "symptom": "intermittent TimeoutError in test_retry_after_soft_decline under CI load",
        "objective": "find why the soft-decline retry test intermittently exceeds the harness timeout",
        "field": "timeout_ms",
        "kind": "int",
        "stale": "250",
        "new": "1750",
        "evidence_cmd": "pytest -q tests/integration/test_checkout.py --showlocals",
        "evidence_source": "tests/conftest.py::pytest_configure (post-change dump)",
    },
    {
        "sid": "s02_worker_pool",
        "domain": "background job processing",
        "component": "celery worker pool",
        "repo": "acme/ledger-worker",
        "service": "ledger-worker",
        "symbol": "process_batch",
        "paths": ["worker/celery_app.py", "worker/tasks.py", "deploy/worker.env", "docker-compose.yml"],
        "commands": ["celery -A worker.celery_app inspect active", "celery -A worker.celery_app inspect stats", "grep -R worker_pool deploy"],
        "symptom": "queue depth grows and batches are deferred under moderate load",
        "objective": "determine the effective worker pool size after the deployment change",
        "field": "worker_pool",
        "kind": "int",
        "stale": "4",
        "new": "16",
        "evidence_cmd": "celery -A worker.celery_app inspect stats",
        "evidence_source": "worker stats dump, pool section",
    },
    {
        "sid": "s03_tls_error_code",
        "domain": "edge proxy diagnostics",
        "component": "ingress gateway upstream TLS",
        "repo": "acme/platform-infra",
        "service": "edge-gateway",
        "symbol": "establish_upstream",
        "paths": ["deploy/gateway.yaml", "internal/tls/client.go", "logs/gateway.log", "config/upstreams.yaml"],
        "commands": ["kubectl -n edge logs deploy/edge-gateway --since=15m", "kubectl -n edge get pods -l app=edge-gateway", "openssl s_client -connect upstream.internal:443"],
        "symptom": "intermittent 502 responses during upstream TLS handshakes",
        "objective": "identify the authoritative post-change TLS error code on the upstream path",
        "field": "error_code",
        "kind": "token",
        "stale": "E_CONN_RESET",
        "new": "E_TLS_HANDSHAKE_TIMEOUT",
        "evidence_cmd": "kubectl -n edge logs deploy/edge-gateway --since=15m",
        "evidence_source": "edge-gateway structured log, final handshake record",
    },
    {
        "sid": "s04_db_pool",
        "domain": "PostgreSQL connection pooling",
        "component": "checkout API database layer",
        "repo": "acme/checkout-service",
        "service": "checkout-api",
        "symbol": "acquire_session",
        "paths": ["src/checkout/db.py", "config/production.toml", "migrations/versions/0042_pool.sql", "docs/runbooks/db.md"],
        "commands": ["psql -c 'show max_connections'", "psql -c 'select count(*) from pg_stat_activity'", "pgbouncer -R -d config.ini"],
        "symptom": "database pool wait time spikes during peak checkout traffic",
        "objective": "confirm the new database pool ceiling applied by the rolling deploy",
        "field": "db_pool_max",
        "kind": "int",
        "stale": "12",
        "new": "48",
        "evidence_cmd": "psql -Atc 'show db_pool_max' || true; env | grep DB_POOL",
        "evidence_source": "runtime configuration dump from the new checkout pod",
    },
    {
        "sid": "s05_k8s_probe",
        "domain": "Kubernetes rollout health",
        "component": "checkout deployment readiness probe",
        "repo": "acme/platform-infra",
        "service": "checkout-api",
        "symbol": "readiness_handler",
        "paths": ["k8s/checkout/deployment.yaml", "src/health.py", "k8s/checkout/service.yaml", "docs/runbooks/rollout.md"],
        "commands": ["kubectl -n prod describe deploy checkout-api", "kubectl -n prod get events --sort-by=.lastTimestamp", "kubectl -n prod rollout status deploy/checkout-api"],
        "symptom": "pods briefly flap during rolling updates before becoming Ready",
        "objective": "find the effective readiness failureThreshold after the manifest change",
        "field": "failure_threshold",
        "kind": "int",
        "stale": "3",
        "new": "7",
        "evidence_cmd": "kubectl -n prod get deploy checkout-api -o json",
        "evidence_source": "deployment spec, readinessProbe section",
    },
    {
        "sid": "s06_api_retry",
        "domain": "rate-limited JSON API client",
        "component": "vendor sync client",
        "repo": "acme/sync-jobs",
        "service": "vendor-sync",
        "symbol": "backoff_delay",
        "paths": ["src/sync/vendor_client.py", "tests/test_vendor_client.py", "config/sync.yaml", "docs/vendor-api.md"],
        "commands": ["python -m pytest tests/test_vendor_client.py -q", "curl -s -D - https://vendor.example/v1/limits", "grep -R retry_after config src"],
        "symptom": "429 responses cause sync gaps because retries resume before the vendor window resets",
        "objective": "determine the current retry-after delay enforced by the vendor client",
        "field": "retry_after_ms",
        "kind": "int",
        "stale": "500",
        "new": "2750",
        "evidence_cmd": "python -m src.sync.vendor_client --dump-effective-config",
        "evidence_source": "effective client config, retry section",
    },
    {
        "sid": "s07_systemd_restart",
        "domain": "systemd service supervision",
        "component": "metrics collector service",
        "repo": "acme/observability-stack",
        "service": "metrics-collector",
        "symbol": "ExecStartPre",
        "paths": ["/etc/systemd/system/metrics-collector.service", "scripts/collector.sh", "config/collector.yaml", "logs/collector.log"],
        "commands": ["systemctl status metrics-collector --no-pager", "journalctl -u metrics-collector -n 200 --no-pager", "systemctl show metrics-collector -p StartLimitBurst"],
        "symptom": "a startup crash loop trips start-limit-hit and leaves the collector down",
        "objective": "confirm the restarted service's burst limit in the current unit file",
        "field": "restart_burst",
        "kind": "int",
        "stale": "5",
        "new": "20",
        "evidence_cmd": "systemctl show metrics-collector -p StartLimitBurst -p Restart",
        "evidence_source": "loaded systemd unit (post-daemon-reload)",
    },
    {
        "sid": "s08_nginx_timeout",
        "domain": "reverse proxy tuning",
        "component": "nginx upstream for reports",
        "repo": "acme/web-edge",
        "service": "api-edge",
        "symbol": "proxy_pass",
        "paths": ["/etc/nginx/conf.d/api.conf", "logs/nginx/error.log", "config/upstreams.conf", "deploy/edge-values.yaml"],
        "commands": ["nginx -T | sed -n '/reports/,/}/p'", "tail -n 200 /var/log/nginx/error.log", "curl -s -o /dev/null -w '%{http_code} %{time_total}' https://api.example/reports/export"],
        "symptom": "long report exports intermittently return 504 from the edge proxy",
        "objective": "determine the effective read timeout on the reports upstream",
        "field": "proxy_read_timeout_s",
        "kind": "int",
        "stale": "30",
        "new": "95",
        "evidence_cmd": "nginx -T | grep -A12 'upstream reports_backend'",
        "evidence_source": "rendered nginx configuration after reload",
    },
    {
        "sid": "s09_cert_renewal",
        "domain": "certificate rotation",
        "component": "cert-manager renewal window",
        "repo": "acme/platform-infra",
        "service": "cert-manager",
        "symbol": "RenewBefore",
        "paths": ["deploy/certificate.yaml", "scripts/rotate_cert.sh", "docs/runbooks/tls.md", "k8s/cert-manager/values.yaml"],
        "commands": ["kubectl -n cert-manager describe certificate api-tls", "kubectl -n cert-manager get certificaterequests", "openssl x509 -in /tmp/api.crt -noout -dates"],
        "symptom": "renewal happens too late during a maintenance freeze and risks expiry",
        "objective": "find the renewal lead time applied to the api-tls certificate",
        "field": "renew_before_days",
        "kind": "int",
        "stale": "14",
        "new": "45",
        "evidence_cmd": "kubectl -n cert-manager get certificate api-tls -o yaml",
        "evidence_source": "Certificate resource spec after reconciliation",
    },
    {
        "sid": "s10_redis_eviction",
        "domain": "cache layer operations",
        "component": "Redis session cache",
        "repo": "acme/checkout-service",
        "service": "redis-sessions",
        "symbol": "eviction_policy",
        "paths": ["infra/redis.conf", "src/cache/redis_client.py", "docs/cache.md", "deploy/redis-statefulset.yaml"],
        "commands": ["redis-cli CONFIG GET maxmemory-policy", "redis-cli INFO memory | head -30", "redis-cli INFO stats | grep evicted"],
        "symptom": "cache-miss storms coincide with session keys disappearing under memory pressure",
        "objective": "confirm the current maxmemory eviction policy on the session cache",
        "field": "maxmemory_policy",
        "kind": "token",
        "stale": "noeviction",
        "new": "allkeys-lfu",
        "evidence_cmd": "redis-cli CONFIG GET maxmemory-policy",
        "evidence_source": "live Redis CONFIG GET response",
    },
]

NOTE_TEMPLATES = [
    "Repository scan: {path} contains the {symbol} entry point; the call chain enters from the runner and leaves through the retry wrapper. No duplicate symbol definitions were found in this pass.",
    "Command `{cmd}` exited with status {status} after {secs:.2f}s. The log was truncated to the last {lines} lines; the first anomaly appears after the retry budget is exhausted.",
    "Static inspection of {path} shows two configuration readers with overlapping precedence rules. The later reader wins at runtime, which explains why the checked-in default can disagree with observed behaviour.",
    "The most recent failing run re-entered {symbol} from the scheduler rather than from the request handler. Frame numbers in the traceback should not be treated as call order.",
    "A grep across the workspace found {count} references to {symbol}; {count2} of them are tests, and the remaining references are wiring or documentation.",
    "We compared the last two deploy snapshots. The diff is small but touches the startup path, so a restart is required before trusting any cached process state.",
    "Local reproduction with `{cmd}` is intermittent: {ok} of {total} attempts passed. The failures cluster when the host is under load, not when the input is malformed.",
    "The service log rotates quickly. We filtered to the last {lines} lines and saw repeated retry scheduling with increasing delays, then a reset to the default interval.",
    "Hypothesis {hyp}: the reported symptom is caused by a stale value captured before the rollout, not by the code path currently under test. Needs post-change evidence.",
    "Path {path} has no local unit tests for the failure branch. Coverage is indirect through the integration suite, so the absence of a unit failure is not evidence of correctness.",
    "One open question is whether the runner reads configuration at process start or per request. If it is per request, an old worker could still serve a stale value briefly.",
    "We inspected process environment, mounted config, and command-line overrides. The precedence order appears to be: CLI overrides, then environment, then mounted config, then compiled defaults.",
    "The trace includes {count} retries and {warn} warnings. None of the warnings is fatal, but together they inflate the observed latency enough to look like a timeout.",
    "A diff against the previous release shows only comments and logging changes in {path}; no behavioural change should be inferred from that file alone.",
    "The scheduler queue was empty during the failing window, which rules out head-of-line blocking as the primary cause. Thread-pool saturation is still possible.",
    "We captured a heap-free profile of the runner. Allocation is flat; the long pole is waiting on the external call, so the next step is to inspect its effective timeout.",
    "Timestamp correlation: the first 502 in {path2} lands {delta_ms} ms after a successful health check. That is consistent with a handshake or connection-reset path.",
    "The repository has a feature flag guarding this path. We confirmed the flag is enabled in the failing environment; the disabled path is irrelevant to the current symptom.",
    "A byte-level diff of the generated config shows whitespace-only changes plus one numeric override. Numeric overrides are the usual source of stale-snapshot confusion.",
    "Before the change, the runner logged a compact one-line summary per attempt. After the change it logs structured JSON, so old dashboards may be parsing the wrong fields.",
    "We re-ran the narrowest test three times with the same seed. Two runs passed; one failed. The failure is load-sensitive rather than data-sensitive.",
    "The relevant timeout is read through a helper that caches the value for the lifetime of the worker. A hot reload alone will not refresh it; a process restart is required.",
    "There are two candidate symbols with similar names in {path}. The traceback resolves to {symbol}; the other one is dead code retained for an old migration.",
    "The system journal shows a clean start followed by a restart burst. No OOM kill was recorded, so the exit path is likely an application-level guard.",
    "We verified that DNS resolution was healthy during the incident window. The upstream error was therefore in the TLS / connection phase, not address lookup.",
    "The last deploy changed the manifest and the runtime config in the same rollout. That makes it hard to distinguish a code regression from a configuration regression.",
    "A quick calculation from the log timestamps suggests the effective retry window is longer than the stale summary claims. This must be revalidated from fresh output.",
    "We searched for the old default in the repository. It appears in documentation and in a test fixture, but not in the currently rendered runtime config.",
    "The service exposes a diagnostic endpoint that dumps its effective settings. That dump is more trustworthy than the checked-in YAML after a rollout.",
    "Log sampling is enabled at 25%. A missing line in the sampled stream is not strong evidence; the structured counters are more reliable for this investigation.",
    "The runner keeps a per-session cache of tool results. This summary is the compacted form of those results; values captured before the latest tool call are provisional.",
    "No changes were made to {path} in the last deploy. If the symptom persists, focus on the runtime environment rather than on source control history.",
    "We confirmed the working tree is clean and the build matches the deployed image digest. Local edits are not confounding the reproduction.",
    "The failing window coincides with a burst of retries from a different worker. Cross-worker contention could inflate latency, but should not change the configured value.",
    "The previous summary was generated by an older compaction pass. Its wording is compressed and some provisional numbers may no longer reflect the live system.",
    "We inspected the resource limits: CPU and memory are both below their ceilings. The remaining suspect is a wait or timeout configuration, not resource exhaustion.",
    "A counterfactual run with the old binary passed once and failed once. This is too noisy to attribute the regression, so the investigation is still open.",
    "The error payload includes a request identifier that we can join against the gateway log. The join succeeded for the last two failures and points to the same upstream.",
    "We reviewed the runbook. It still documents the pre-change value and should be treated as documentation drift until the post-change evidence is inspected.",
    "The checked-in default in {path} is overridden by the deployment environment. Reading the file alone would give the wrong conclusion.",
    "No preemption or eviction was reported by the runtime during the failing window. The symptom is more consistent with a configuration mismatch than with memory pressure.",
    "We captured the effective command line of the running process. It includes an override that is not present in the repository, which explains the earlier confusion.",
    "The traceback is truncated after the first cause chain. The nested cause is the informative part; the top-level exception is only a wrapper.",
    "A comparison of healthy and failing pods shows the same image and the same node pool. The difference is in the injected configuration snapshot.",
    "The deployment controller reports the rollout as complete, but the old ReplicaSet was still serving traffic during the failing window. Stale pods are a plausible confounder.",
    "We ran the diagnostic command against a canary instance. It responded quickly and with the expected schema; the production path is the one that needs fresh evidence.",
    "The log line that would settle the question is emitted only at startup. A rolling restart is required before that line reflects the new value.",
    "We checked for hidden Unicode or trailing whitespace in {path}. The parser is strict, but no malformed keys were found in this pass.",
    "The test harness retries failures automatically, which inflates the apparent flake rate. Counts in this summary are per attempt, not per test case.",
    "We inspected the upstream health checker. It uses a shorter timeout than the data path, so a slow handshake can look healthy at probe time and fail later.",
    "The current hypothesis is {hyp}. It is still provisional; the definitive answer must come from the post-change tool result, not from this working memory.",
    "A local reproduction under a CPU stressor reproduced the symptom once in five runs. The strongest signal is still the effective configuration value.",
    "We compared the rendered config on two replicas. They agree, which suggests the rollout propagated consistently; a per-pod drift is unlikely.",
    "The service's restart policy is `on-failure`. A configuration error at startup will therefore loop rather than stay down, making the failure look intermittent.",
    "No schema migration ran in this deploy. The issue is unlikely to be a database compatibility problem, although the pool settings could still be wrong.",
    "We confirmed that the session cache is reachable and serving reads. The interesting question is the eviction policy, which determines what happens under pressure.",
    "The gap between the stale snapshot and the current process state is the main risk in this investigation. Re-read the latest tool evidence before concluding.",
    "The package version reported by the running binary matches the lockfile. Dependency drift is not a leading hypothesis for this symptom.",
    "We saw the same warning in the previous incident, where it was benign. Do not treat the warning alone as causal evidence.",
    "The latency histogram has a long tail starting at the retry boundary. That pattern is consistent with retry amplification, but only the effective timeout can confirm it.",
    "We checked the scheduler's clock skew: under 2 ms across nodes. Timestamp ordering within the same second should be trustworthy.",
    "The failure reproduces with a minimal fixture that does not touch the external network. This weakens the upstream-service hypothesis and strengthens a local-timeout hypothesis.",
    "The runner's working directory contains a stale `.env` file from an earlier session. We confirmed it is not loaded by the production entry point, but it can confuse local repros.",
    "We inspected the compiled configuration object at runtime. It reports its source as the environment layer, not the checked-in defaults.",
    "The code path is covered by an integration test, but that test asserts only the exit status, not the effective timeout. A regression in the value would go unnoticed.",
    "The operator log records a configuration reload shortly before the incident. The reload may have partial effect if some workers cache values at startup.",
    "We have not yet inspected the post-change diagnostic dump. Until then, every numeric value in this summary is provisional and must not be treated as ground truth.",
    "A grep for the field name in the repository finds defaults, tests, and docs. None of those reflects the live process state after the override was applied.",
    "We verified the container image digest against the release manifest. The image is correct, so the discrepancy is in configuration, not build provenance.",
    "The scheduler backoff and the upstream retry-after are separate knobs. Conflating them is a common cause of wrong conclusions in this area.",
    "One replica was restarted manually during triage. Its effective configuration may differ from the others until the rollout converges, so replica-level samples need care.",
    "The trace shows a clean handshake on the first attempt and a reset on the second. That asymmetry suggests connection reuse or a TLS session issue rather than a static timeout.",
    "We inspected the process start time. It predates the configuration change, which is strong evidence that the process has not picked up the new value yet.",
    "The documentation still describes the old capacity plan. The live configuration is the only source of truth for this task.",
    "No rate limiting was observed at the ingress during the failing window. The upstream 429s originated beyond our edge and are outside the local retry budget.",
    "We checked the kernel TCP counters. There were retransmits but no socket exhaustion, so the failure is more likely a policy timeout than a resource limit.",
    "The failing test uses a fixture that sleeps before asserting. Under load, the sleep and the harness timeout race, which makes the observed failure look random.",
    "We reviewed the last three incident reports. Two mention stale config after a rollout; one mentions a genuine code bug. The current evidence pattern resembles the former.",
    "The service logs its effective values only at INFO level, and the production level is WARN. That is why the startup dump is missing from normal logs and why the diagnostic command is needed.",
    "A fresh diagnostic request is pending. The runner has not yet attached its output; do not finalize a value based on this compacted history.",
    "We confirmed the change window and the failing window overlap. Correlation is not causation, but it raises the priority of checking effective runtime values.",
    "The old summary states a provisional value. The task is specifically to replace it with the authoritative post-change value from the next tool result.",
    "We validated that the workspace is the expected repository and branch. Path confusion is unlikely to explain the mismatch.",
    "The scheduler has a global concurrency cap that was unchanged. It is therefore not a plausible explanation for a per-field configuration symptom.",
    "We inspected the serialized config for duplicate keys. None were found; the parser is deterministic, and the last layer wins as documented.",
    "The latest tool call is intentionally diagnostic. Its output should contain a single authoritative statement that resolves the remaining ambiguity.",
    "We have enough context to know where to look, but not enough to answer. The next step is to read the post-change tool result rather than infer from the summary.",
    "The trace shows the failure occurring before the first successful health check. That ordering favours a startup-time configuration problem over a runtime data problem.",
    "We compared the effective values on the canary and production. They differ, which is expected during a staged rollout and reinforces the need for post-change evidence.",
    "The repository's CI uses a shorter timeout than production. A green CI run does not prove that the production value is correct.",
    "We checked the configuration mount's inode timestamp. It changed during the deploy, so the file on disk is newer than the running process in at least one replica.",
    "The log message includes both the requested value and the effective value. We need the post-change dump to see which one the current process actually uses.",
    "No obvious memory leak was found. Resident memory is stable across the failing window, which argues against eviction or OOM as the root cause.",
    "We reviewed the test's retry decorator. It can mask a slow dependency by retrying, but it cannot mask a misconfigured timeout.",
    "The field in question is consumed by a library that parses it once at construction time. Later edits to the environment will not affect existing workers.",
    "A read-only diagnostic was dispatched. Its output is not yet attached; the runner will include it in the next turn and it overrides older observations.",
    "We checked the upstream vendor status page. No incident was reported during the window, so a local retry-policy mismatch remains the leading hypothesis.",
    "The old value appears in a cached runtime object. That is exactly the stale state this investigation needs to distinguish from the current on-disk configuration.",
    "We inspected the controller reconciliation logs. The desired state was applied successfully, but pod-level observation is still needed to confirm the effective value.",
    "The final answer must be one line and must use the new value from the tool output. Do not include the stale value anywhere in the final answer.",
    "We have repeated this check across two sessions and gotten the same provisional snapshot. Repetition of a stale value is not confirmation; it is the same stale source.",
    "The next tool result is the authoritative source. Treat every prior numeric value as a hypothesis until that result is attached.",
]

HYPOTHESES = [
    "H1 stale config after rollout",
    "H2 retry budget amplification",
    "H3 connection reuse race",
    "H4 pool saturation",
    "H5 rollout partial reload",
    "H6 log sampling artefact",
    "H7 probe timeout mismatch",
    "H8 environment precedence inversion",
    "H9 stale worker process",
    "H10 upstream handshake regression",
]


@dataclass
class Scenario:
    sid: str
    domain: str
    component: str
    repo: str
    service: str
    symbol: str
    paths: List[str]
    commands: List[str]
    symptom: str
    objective: str
    field_name: str
    kind: str
    stale: str
    new: str
    evidence_cmd: str
    evidence_source: str
    spec: Dict[str, Any]
    notes: List[str] = field(default_factory=list)
    tool_text: str = ""
    tool_text_tokens: int = 0


def _rng_for(cfg: Dict[str, Any], *parts: int) -> random.Random:
    seed = int(cfg["benchmark"]["seed"])
    mixed = seed * 1_000_003
    for p in parts:
        mixed = (mixed * 10_007 + p * 101) & 0x7FFFFFFF
    return random.Random(mixed)


def make_notes(spec: Dict[str, Any], rng: random.Random, count: int) -> List[str]:
    notes: List[str] = []
    hypotheses = list(HYPOTHESES)
    for i in range(count):
        template = NOTE_TEMPLATES[i % len(NOTE_TEMPLATES)]
        path = spec["paths"][i % len(spec["paths"])]
        path2 = spec["paths"][(i + 2) % len(spec["paths"])]
        cmd = spec["commands"][i % len(spec["commands"])]
        count_refs = rng.randint(3, 90)
        notes.append(
            template.format(
                path=path,
                path2=path2,
                cmd=cmd,
                symbol=spec["symbol"],
                status=rng.choice([0, 1, 2, 124, 137]),
                secs=rng.uniform(0.2, 95.0),
                lines=rng.choice([40, 80, 120, 200, 400, 800]),
                count=count_refs,
                count2=rng.randint(0, min(12, count_refs)),
                ok=rng.randint(1, 4),
                total=5,
                warn=rng.randint(1, 9),
                delta_ms=rng.randint(30, 900),
                hyp=rng.choice(hypotheses),
            )
        )
    return notes

def summary_head_original(spec: Scenario) -> str:
    return (
        "=== EARLIER CONTEXT / WORKING SUMMARY - compaction v1 ===\n"
        f"Objective: {spec.objective}.\n"
        f"Repository / service: {spec.repo} / {spec.service}.\n"
        f"Component under investigation: {spec.component}.\n"
        f"Observed symptom: {spec.symptom}.\n"
        "Earlier investigation status: two preliminary tool calls have run; the runner compacted its "
        "working memory at step 41. No post-change diagnostic has been attached yet.\n"
        f"PRELIMINARY SNAPSHOT (step 12, NOT revalidated): {spec.field_name}={spec.stale}. "
        "Treat this as stale context until a post-change tool result arrives.\n"
        "Open task: inspect the next post-change tool result, reconcile it with the earlier snapshot, "
        "and answer with the authoritative current value. Do not infer from this summary if the tool "
        "result disagrees.\n"
        "Constraints: do not guess; cite only the latest tool evidence; keep the final answer on one line.\n"
        f"Workspace: {spec.repo}.\n"
        f"Active symbol: {spec.symbol}.\n"
        f"Likely files: {', '.join(spec.paths[:3])}.\n"
        "------------------------------------------------------------------\n"
        "DETAILED CONTEXT NOTES (chronological, compacted):\n"
    )


def summary_head_rewritten(spec: Scenario) -> str:
    return (
        "=== WORKING MEMORY - compaction v2 (reworded by the runner, same facts) ===\n"
        "This block restates the earlier findings with different wording. No factual state has changed.\n"
        f"Goal: {spec.objective}.\n"
        f"Deployment: {spec.service} inside {spec.repo}.\n"
        f"Area: {spec.component}.\n"
        f"Failure pattern observed so far: {spec.symptom}.\n"
        "Progress: the runner completed two preliminary tool calls and compacted its notes at step 41; "
        "the post-change diagnostic is still pending.\n"
        f"STALE PRE-CHANGE OBSERVATION (step 12, awaiting revalidation): {spec.field_name}={spec.stale}.\n"
        "The next post-change tool result is authoritative. If it conflicts with this stale observation, "
        "the tool result wins.\n"
        "Answer discipline: one line, exact value, no speculation.\n"
        f"Working tree: {spec.repo}; focus symbol: {spec.symbol}; candidate files: {', '.join(spec.paths[:3])}.\n"
        "------------------------------------------------------------------\n"
        "DETAILED CONTEXT NOTES (chronological, compacted):\n"
    )


def summary_head_compressed(spec: Scenario) -> str:
    return (
        "=== COMPACT WORKING MEMORY - compression v3 ===\n"
        f"Goal: {spec.objective}.\n"
        f"Service {spec.service} ({spec.repo}), component {spec.component}; "
        f"symptom: {spec.symptom}.\n"
        f"Stale pre-change snapshot (step 12): {spec.field_name}={spec.stale}. "
        "Await the next tool result; it overrides this value.\n"
        f"Focus: {spec.symbol} in {', '.join(spec.paths[:2])}. Answer on one line from the latest evidence.\n"
        "------------------------------------------------------------------\n"
        "CONDENSED NOTES:\n"
    )


def task_message(spec: Scenario) -> str:
    return (
        f"Current objective: {spec.objective}.\n"
        f"You are mid-trajectory on {spec.repo} / {spec.service}. The earlier-context summary above was "
        "compacted at step 41. The last action was a tool call that requested a fresh post-change "
        f"diagnostic for {spec.component}. The diagnostic output has not been attached yet.\n"
        "Do not finalize an answer from the stale summary. When the post-change tool result arrives, "
        f"reconcile it and report the authoritative value of `{spec.field_name}`.\n"
        "Constraints: answer budget is short, do not invent values, and do not answer before the tool "
        "result is present."
    )


def assistant_message(spec: Scenario) -> str:
    return (
        f"I will collect the post-change evidence for {spec.component} before drawing a conclusion.\n"
        "Tool call: shell\n"
        "```json\n"
        f'{{"name": "shell", "arguments": {{"command": "{spec.evidence_cmd}", "timeout_ms": 120000}}}}\n'
        "```\n"
        "The earlier snapshot is treated as stale until this result arrives. Waiting for the tool runner "
        "to attach the output."
    )


def pending_user_message(spec: Scenario) -> str:
    return (
        f"TOOL RUNNER: The requested post-change output for {spec.component} will be attached in the next "
        "message. Acknowledge and wait; do not answer yet."
    )


def system_content(salt: str) -> str:
    return (
        f"SESSION-ID: {salt}\n"
        "You are a senior software and operations agent running on a local workstation. You debug real "
        "repositories and services using tools, then report only what the latest tool evidence supports.\n\n"
        "Available tools (text protocol; no native tool channel is required):\n"
        "- shell: run a shell command in the workspace; returns stdout, stderr, exit_code.\n"
        "- read_file: read a file by path; optional line range.\n"
        "- grep: search a repository for a literal string or regex.\n\n"
        "Behavioural rules:\n"
        "1. Treat compacted summaries as provisional. Any value marked stale must be revalidated.\n"
        "2. The most recent tool output is the authoritative source for values it explicitly reports.\n"
        "3. Never invent identifiers, error codes, configuration values, or log lines.\n"
        "4. When asked for a final answer, emit exactly one final line and no extra commentary.\n"
        "5. Keep the answer budget short; precision matters more than volume.\n"
        "6. If the evidence is not yet attached, wait instead of guessing."
    )


def tool_instruction(spec: Scenario) -> str:
    return (
        "Now answer using ONLY the latest post-change tool output above. State the current authoritative "
        f"value of `{spec.field_name}`.\n"
        "Reply with exactly one line in this format:\n"
        f"FINAL: {spec.field_name}=VALUE\n"
        "Replace VALUE with the exact value reported by the post-change evidence. Do not mention the stale "
        "snapshot or any earlier value."
    )


def _log_lines(spec: Scenario, rng: random.Random) -> List[str]:
    templates = [
        "2026-11-21T09:{mm:02d}:{ss:02d}.{ms:03d}Z INFO  pid={pid} unit={unit} attempt={attempt} state={state} elapsed_ms={elapsed}",
        "2026-11-21T09:{mm:02d}:{ss:02d}.{ms:03d}Z DEBUG pid={pid} unit={unit} frame={frame} symbol={symbol} step={step}",
        "2026-11-21T09:{mm:02d}:{ss:02d}.{ms:03d}Z INFO  pid={pid} unit={unit} probe={probe} result={result} latency_ms={elapsed}",
        "2026-11-21T09:{mm:02d}:{ss:02d}.{ms:03d}Z WARN  pid={pid} unit={unit} retry={attempt} reason=transient backoff_ms={backoff}",
        "2026-11-21T09:{mm:02d}:{ss:02d}.{ms:03d}Z INFO  pid={pid} unit={unit} checkpoint={frame} bytes={bytes_} digest={digest}",
        "2026-11-21T09:{mm:02d}:{ss:02d}.{ms:03d}Z DEBUG pid={pid} unit={unit} config_layer={layer} key={key} precedence={precedence}",
        "2026-11-21T09:{mm:02d}:{ss:02d}.{ms:03d}Z INFO  pid={pid} unit={unit} upstream={upstream} handshake={handshake} reuse={reuse}",
        "2026-11-21T09:{mm:02d}:{ss:02d}.{ms:03d}Z TRACE pid={pid} unit={unit} queue_depth={depth} workers_busy={busy} workers_total={total}",
        "2026-11-21T09:{mm:02d}:{ss:02d}.{ms:03d}Z INFO  pid={pid} unit={unit} cache={cache} hit_ratio={ratio:.3f} evictions={evictions}",
        "2026-11-21T09:{mm:02d}:{ss:02d}.{ms:03d}Z DEBUG pid={pid} unit={unit} file={path} mtime={mtime} size={size}",
    ]
    units = ["scheduler", "probe", "upstream", "runner", "config", "cache", "exec", "watchdog"]
    states = ["pending", "running", "waiting", "retrying", "ready", "draining"]
    layers = ["defaults", "mounted-config", "env", "cli-override"]
    keys = ["retry", "timeouts", "pool", "probes", "tls", "limits", "restart"]
    upstreams = ["ledger", "vendor", "reports", "auth", "billing", "search"]
    handshakes = ["ok", "ok", "ok", "reused", "timeout", "reset"]
    lines: List[str] = []
    for i in range(200):
        lines.append(
            templates[i % len(templates)].format(
                mm=rng.randint(0, 59),
                ss=rng.randint(0, 59),
                ms=rng.randint(0, 999),
                pid=rng.randint(1000, 9999),
                unit=units[i % len(units)],
                attempt=rng.randint(1, 6),
                state=states[i % len(states)],
                elapsed=rng.randint(1, 9000),
                frame=rng.randint(1, 400),
                symbol=spec.symbol,
                step=rng.randint(1, 80),
                probe=rng.choice(["startup", "readiness", "liveness", "deep"]),
                result=rng.choice(["pass", "pass", "pass", "fail", "degraded"]),
                backoff=rng.choice([50, 100, 250, 500, 1000, 2000]),
                bytes_=rng.randint(512, 65536),
                digest=f"{rng.getrandbits(32):08x}",
                layer=layers[i % len(layers)],
                key=keys[i % len(keys)],
                precedence=rng.randint(1, 4),
                upstream=upstreams[i % len(upstreams)],
                handshake=handshakes[i % len(handshakes)],
                reuse=rng.choice(["true", "true", "false"]),
                depth=rng.randint(0, 900),
                busy=rng.randint(0, 32),
                total=rng.randint(4, 32),
                cache=rng.choice(["session", "config", "prepared", "dns"]),
                ratio=rng.uniform(0.1, 0.99),
                evictions=rng.randint(0, 40),
                path=spec.paths[i % len(spec.paths)],
                mtime=f"2026-11-21T09:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z",
                size=rng.randint(200, 90000),
            )
        )
    return lines


def build_tool_text(tokenizer: Any, spec: Scenario, target_tokens: int, rng: random.Random) -> Tuple[str, int]:
    lines = _log_lines(spec, rng)
    header = (
        "=== POST-CHANGE TOOL RESULT ===\n"
        "tool: shell\n"
        f"command: {spec.evidence_cmd}\n"
        "exit_code: 0\n"
        "started_at: 2026-11-21T09:14:02Z\n"
        "--- begin raw output ---\n"
    )
    evidence = (
        "\n=== AUTHORITATIVE POST-CHANGE EVIDENCE (overrides earlier snapshot) ===\n"
        f"{spec.field_name}={spec.new}\n"
        f"source: {spec.evidence_source}\n"
        "confidence: confirmed\n"
        "note: this value supersedes all pre-change snapshots.\n"
        "=== END TOOL RESULT ===\n"
    )

    def text_with(k: int, pad: int) -> str:
        body = "\n".join(lines[:k])
        pad_text = "\n".join(
            f"2026-11-21T09:14:5{idx % 10}.000Z INFO  pid=4242 unit=watchdog heartbeat=ok seq={idx}"
            for idx in range(pad)
        )
        return header + body + (("\n" + pad_text) if pad_text else "") + evidence

    lo, hi, best_k = 0, len(lines), 0
    while lo <= hi:
        mid = (lo + hi) // 2
        n = len(tokenizer.encode(text_with(mid, 0), add_special_tokens=False))
        if n <= target_tokens:
            best_k, lo = mid, mid + 1
        else:
            hi = mid - 1
    best = (abs(len(tokenizer.encode(text_with(best_k, 0), add_special_tokens=False)) - target_tokens), 0)
    for pad in range(0, 40):
        n = len(tokenizer.encode(text_with(best_k, pad), add_special_tokens=False))
        diff = abs(n - target_tokens)
        if diff < best[0]:
            best = (diff, pad)
    text = text_with(best_k, best[1])
    return text, len(tokenizer.encode(text, add_special_tokens=False))


def build_scenarios(tokenizer: Any, cfg: Dict[str, Any]) -> List[Scenario]:
    n = int(cfg["benchmark"]["num_scenarios"])
    tool_target = int(cfg["benchmark"]["tool_output_target_tokens"])
    scenarios: List[Scenario] = []
    for idx, raw in enumerate(SCENARIO_SPECS[:n]):
        sc = Scenario(
            sid=raw["sid"],
            domain=raw["domain"],
            component=raw["component"],
            repo=raw["repo"],
            service=raw["service"],
            symbol=raw["symbol"],
            paths=list(raw["paths"]),
            commands=list(raw["commands"]),
            symptom=raw["symptom"],
            objective=raw["objective"],
            field_name=raw["field"],
            kind=raw["kind"],
            stale=str(raw["stale"]),
            new=str(raw["new"]),
            evidence_cmd=raw["evidence_cmd"],
            evidence_source=raw["evidence_source"],
            spec=dict(raw),
        )
        sc.notes = make_notes(dict(raw), _rng_for(cfg, idx, 11), 180)
        sc.tool_text, sc.tool_text_tokens = build_tool_text(
            tokenizer, sc, tool_target, _rng_for(cfg, idx, 23)
        )
        scenarios.append(sc)
    return scenarios

# =============================================================================
# Prompt construction with token-exact fitting
# =============================================================================

@dataclass
class ArmPrompt:
    arm: str
    salt: str
    prime_messages: List[Dict[str, Any]]
    measured_messages: List[Dict[str, Any]]
    prime_tokens: int
    total_tokens: int
    summary_text: str
    summary_note_count: int
    summary_pad_units: int


@dataclass
class SamplePrompts:
    sample_id: str
    scenario: Scenario
    rep: int
    arms: Dict[str, ArmPrompt]
    checks: Dict[str, Any]


class PromptFactory:
    """Builds token-accurate, trial-salted Turn-1 / Turn-2 prompt pairs."""

    def __init__(self, tokenizer: Any, cfg: Dict[str, Any]):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.prefix_target = int(cfg["benchmark"]["prefix_target_tokens"])
        self.parity_tol = int(cfg["benchmark"]["token_parity_tolerance"])
        self.shorter_ratio = float(cfg["benchmark"]["shorter_prompt_target_ratio"])
        self.pad_prefix = "\n\nRUNNER CHECKPOINT TRACE (neutral boundary padding): "
        self.summary_tail = "\n--- END OF EARLIER CONTEXT (the runner will attach new evidence later) ---"

    def count(self, messages: List[Dict[str, Any]]) -> int:
        out = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if isinstance(out, dict):
            ids = out["input_ids"]
        elif hasattr(out, "input_ids"):
            ids = out.input_ids
        else:
            ids = out
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        return len(ids)

    def salt(self, scenario_idx: int, rep: int, arm: str, phase: str) -> str:
        rng = _rng_for(
            self.cfg, scenario_idx, rep, ARMS.index(arm), 991 if phase == "measured" else 997
        )
        sid_short = SCENARIO_SPECS[scenario_idx]["sid"].split("_")[0]
        return f"{sid_short}-{rep:02d}-{ARM_CODE[arm]}-{rng.getrandbits(32):08x}"

    def _compose(
        self,
        salt: str,
        summary_text: str,
        suffix: Sequence[Dict[str, Any]],
        scenario: Scenario,
    ) -> List[Dict[str, Any]]:
        messages = [
            {"role": "system", "content": system_content(salt)},
            {"role": "user", "content": summary_text},
            {"role": "user", "content": task_message(scenario)},
            {"role": "assistant", "content": assistant_message(scenario)},
            {"role": "user", "content": pending_user_message(scenario)},
        ]
        messages.extend(suffix)
        return messages

    def _fit_summary(
        self,
        scenario: Scenario,
        salt: str,
        head: str,
        notes: List[str],
        target: int,
        suffix: Sequence[Dict[str, Any]],
        tolerance: int,
    ) -> Tuple[str, int, int, int]:
        """Pick notes + neutral boundary padding so the final prompt hits target."""

        def summary_text(k: int, pad: int) -> str:
            body = "\n".join(notes[:k])
            pad_text = " ok" * pad if pad else ""
            return f"{head}{body}{self.pad_prefix}{pad_text}{self.summary_tail}"

        def measure(k: int, pad: int) -> int:
            return self.count(self._compose(salt, summary_text(k, pad), suffix, scenario))

        base_n = measure(0, 0)
        if base_n > target:
            text = summary_text(0, 0)
            return text, 0, 0, base_n

        lo, hi = 0, len(notes)
        best_k, best_n = 0, base_n
        while lo <= hi:
            mid = (lo + hi) // 2
            n = measure(mid, 0)
            if n <= target:
                best_k, best_n = mid, n
                lo = mid + 1
            else:
                hi = mid - 1

        deficit = target - best_n
        cap = max(240, deficit * 2 + 100)
        lo, hi = 0, cap
        best_pad, best_n = 0, best_n
        while lo <= hi:
            mid = (lo + hi) // 2
            n = measure(best_k, mid)
            if n <= target:
                best_pad, best_n = mid, n
                lo = mid + 1
            else:
                hi = mid - 1

        best = (abs(best_n - target), best_k, best_pad, best_n)
        for pad in range(max(0, best_pad - 8), best_pad + 9):
            n = measure(best_k, pad)
            cand = (abs(n - target), best_k, pad, n)
            if cand < best:
                best = cand

        if best[0] > tolerance:
            for k in sorted({max(0, best_k - 1), min(len(notes), best_k + 1)}):
                n0 = measure(k, 0)
                if n0 > target:
                    continue
                d = target - n0
                guesses = {max(0, d // 2), max(0, d - 2), max(0, d - 1), d, d + 1, d + 2}
                for guess in guesses:
                    for pad in range(max(0, guess - 3), guess + 4):
                        n = measure(k, pad)
                        cand = (abs(n - target), k, pad, n)
                        if cand < best:
                            best = cand

        _, k, pad, n = best
        return summary_text(k, pad), k, pad, n

    def build_sample(self, scenario: Scenario, scenario_idx: int, rep: int, phase: str) -> SamplePrompts:
        sample_id = f"{scenario.sid}-r{rep}"
        head_a = summary_head_original(scenario)
        head_b = summary_head_rewritten(scenario)
        head_c = summary_head_compressed(scenario)
        tool_suffix = [
            {
                "role": "user",
                "content": scenario.tool_text + "\n\n" + tool_instruction(scenario),
            }
        ]

        # 1) Fit the original Turn-1 summary once, using Arm A's salt.
        salt_a = self.salt(scenario_idx, rep, ARM_APPEND, phase)
        summary_a, k_a, pad_a, _ = self._fit_summary(
            scenario, salt_a, head_a, scenario.notes, self.prefix_target, [], 0
        )
        notes_a = scenario.notes[:k_a]

        base_a = self._compose(salt_a, summary_a, [], scenario)
        measured_a = self._compose(salt_a, summary_a, tool_suffix, scenario)
        prime_a = self.count(base_a)
        total_a = self.count(measured_a)

        # 2) Arm A.
        arms: Dict[str, ArmPrompt] = {
            ARM_APPEND: ArmPrompt(
                arm=ARM_APPEND,
                salt=salt_a,
                prime_messages=base_a,
                measured_messages=measured_a,
                prime_tokens=prime_a,
                total_tokens=total_a,
                summary_text=summary_a,
                summary_note_count=k_a,
                summary_pad_units=pad_a,
            )
        }

        # 3) Arm B: same original Turn-1 state primed, then a reworded summary
        #    with total prompt tokens equal to Arm A (+/- tolerance).
        salt_b = self.salt(scenario_idx, rep, ARM_EQUAL, phase)
        prime_b = self._compose(salt_b, summary_a, [], scenario)
        summary_b, k_b, pad_b, _ = self._fit_summary(
            scenario, salt_b, head_b, notes_a, total_a, tool_suffix, self.parity_tol
        )
        measured_b = self._compose(salt_b, summary_b, tool_suffix, scenario)
        total_b = self.count(measured_b)
        arms[ARM_EQUAL] = ArmPrompt(
            arm=ARM_EQUAL,
            salt=salt_b,
            prime_messages=prime_b,
            measured_messages=measured_b,
            prime_tokens=self.count(prime_b),
            total_tokens=total_b,
            summary_text=summary_b,
            summary_note_count=k_b,
            summary_pad_units=pad_b,
        )

        # 4) Arm C: same original Turn-1 state primed, then a compressed summary
        #    targeting ~70% of Arm A's total token count.
        salt_c = self.salt(scenario_idx, rep, ARM_SHORTER, phase)
        prime_c = self._compose(salt_c, summary_a, [], scenario)
        notes_c = notes_a[::2]
        target_c = int(round(self.shorter_ratio * total_a))
        summary_c, k_c, pad_c, _ = self._fit_summary(
            scenario, salt_c, head_c, notes_c, target_c, tool_suffix, 12
        )
        measured_c = self._compose(salt_c, summary_c, tool_suffix, scenario)
        total_c = self.count(measured_c)
        arms[ARM_SHORTER] = ArmPrompt(
            arm=ARM_SHORTER,
            salt=salt_c,
            prime_messages=prime_c,
            measured_messages=measured_c,
            prime_tokens=self.count(prime_c),
            total_tokens=total_c,
            summary_text=summary_c,
            summary_note_count=k_c,
            summary_pad_units=pad_c,
        )

        parity_diff = abs(total_b - total_a)
        shorter_ratio = total_c / max(1, total_a)
        min_ratio = float(self.cfg["benchmark"]["shorter_prompt_min_ratio"])
        max_ratio = float(self.cfg["benchmark"]["shorter_prompt_max_ratio"])
        checks = {
            "arm_a_total_tokens": total_a,
            "arm_b_total_tokens": total_b,
            "arm_c_total_tokens": total_c,
            "parity_abs_diff": parity_diff,
            "parity_within_tolerance": parity_diff <= self.parity_tol,
            "shorter_ratio": shorter_ratio,
            "shorter_in_band": min_ratio <= shorter_ratio <= max_ratio,
            "prefix_target": self.prefix_target,
            "prime_tokens_a": prime_a,
            "prime_tokens_b": arms[ARM_EQUAL].prime_tokens,
            "prime_tokens_c": arms[ARM_SHORTER].prime_tokens,
            "arm_a_notes": k_a,
            "arm_b_notes": k_b,
            "arm_c_notes": k_c,
        }
        return SamplePrompts(
            sample_id=sample_id,
            scenario=scenario,
            rep=rep,
            arms=arms,
            checks=checks,
        )

# =============================================================================
# Server management and telemetry
# =============================================================================

class ServerManager:
    def __init__(self, cfg: Dict[str, Any], root: Path):
        self.cfg = cfg
        self.root = root
        self.s = cfg["vllm_server"]
        self.base_url = f"http://{self.s['host']}:{self.s['port']}"
        self.log_path = ensure_parent(cfg["outputs"]["server_log"], root)
        self.proc: Optional[subprocess.Popen] = None
        self.started_by_us = False
        self.serve_command: List[str] = []
        self.client = httpx.Client(
            timeout=httpx.Timeout(
                connect=10.0,
                read=float(self.s.get("request_timeout_s", 900)),
                write=60.0,
                pool=10.0,
            ),
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
            trust_env=False,
        )

    def healthy(self) -> bool:
        try:
            r = self.client.get(f"{self.base_url}/health", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    def models(self) -> List[str]:
        r = self.client.get(f"{self.base_url}/v1/models", timeout=15.0)
        r.raise_for_status()
        return [m.get("id") for m in r.json().get("data", [])]

    def verify_model(self) -> None:
        ids = self.models()
        expected = {self.s.get("served_model_name"), self.s.get("model")}
        if not expected.intersection(set(ids)):
            raise RuntimeError(f"Server model mismatch: expected {expected}, got {ids}")

    def metrics_snapshot(self) -> Dict[str, float]:
        try:
            r = self.client.get(f"{self.base_url}/metrics", timeout=15.0)
            r.raise_for_status()
            return parse_prometheus(r.text)
        except Exception:
            return {}

    def _vllm_executable(self) -> List[str]:
        exe = shutil.which("vllm")
        if exe:
            return [exe, "serve"]
        return [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]

    def _help_text(self) -> str:
        for flag in ("--help=all", "--help"):
            try:
                cmd = self._vllm_executable() + [flag]
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
                text = (out.stdout or "") + (out.stderr or "")
                if text.strip():
                    return text
            except Exception:
                continue
        return ""

    def build_command(self) -> List[str]:
        s = self.s
        cmd = self._vllm_executable() + [
            str(s["model"]),
            "--host", str(s["host"]),
            "--port", str(s["port"]),
            "--served-model-name", str(s.get("served_model_name") or s["model"]),
            "--dtype", str(s["dtype"]),
            "--max-model-len", str(s["max_model_len"]),
            "--gpu-memory-utilization", str(s["gpu_memory_utilization"]),
        ]
        if s.get("enable_prefix_caching", True):
            cmd.append("--enable-prefix-caching")
        if s.get("trust_remote_code"):
            cmd.append("--trust-remote-code")
        help_text = self._help_text()
        if s.get("enable_prompt_tokens_details", True) and "--enable-prompt-tokens-details" in help_text:
            cmd.append("--enable-prompt-tokens-details")
        if s.get("block_size") and "--block-size" in help_text:
            cmd += ["--block-size", str(int(s["block_size"]))]
        extra = s.get("extra_args") or []
        if isinstance(extra, list):
            cmd += [str(x) for x in extra]
        return cmd

    def start(self) -> None:
        if self.healthy():
            self.verify_model()
            print(f"[server] reusing healthy server at {self.base_url}")
            return
        cmd = self.build_command()
        self.serve_command = cmd
        env = os.environ.copy()
        if self.s.get("hf_home"):
            env["HF_HOME"] = str(self.s["hf_home"])
            env.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(self.s["hf_home"]) / "hub"))
        env.setdefault("VLLM_LOGGING_LEVEL", "INFO")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "wb", buffering=0) as log_fh:
            print(f"[server] starting: {' '.join(cmd)}")
            print(f"[server] log: {self.log_path}")
            self.proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        self.started_by_us = True
        timeout = float(self.s.get("startup_timeout_s", 3600))
        t0 = time.time()
        last_report = 0.0
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited during startup (code {self.proc.returncode}).\n{self.log_tail(50)}"
                )
            if self.healthy():
                self.verify_model()
                print(f"[server] healthy after {time.time() - t0:.1f}s")
                return
            now = time.time()
            if now - last_report > 30:
                last_report = now
                print(f"[server] waiting for /health ... {now - t0:.0f}s | {self.log_last_line()}")
            time.sleep(5)
        raise RuntimeError(f"vLLM did not become healthy within {timeout}s.\n{self.log_tail(80)}")

    def log_tail(self, n: int = 50) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="ignore").splitlines()[-n:])
        except Exception:
            return "<no log>"

    def log_last_line(self) -> str:
        try:
            lines = self.log_path.read_text(errors="ignore").splitlines()
            return lines[-1][:220] if lines else ""
        except Exception:
            return ""

    def stop(self) -> None:
        if self.proc is None or not self.started_by_us or self.proc.poll() is not None:
            return
        print("[server] stopping vLLM")
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except Exception:
            try:
                self.proc.terminate()
            except Exception:
                pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                pass

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


def parse_prometheus(text: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].split("{")[0]
        try:
            value = float(parts[-1])
        except ValueError:
            continue
        values[name] = values.get(name, 0.0) + value
    return values


def pick_metric(snapshot: Dict[str, float], names: Sequence[str]) -> Optional[float]:
    for name in names:
        if name in snapshot:
            return snapshot[name]
    return None


def prefix_metrics(snapshot: Dict[str, float]) -> Tuple[Optional[float], Optional[float]]:
    hits = pick_metric(snapshot, ["vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"])
    queries = pick_metric(snapshot, ["vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"])
    return hits, queries


def preemptions(snapshot: Dict[str, float]) -> Optional[float]:
    return pick_metric(snapshot, ["vllm:num_preemptions_total", "vllm:num_preemptions"])


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def nvidia_smi_query() -> Dict[str, Any]:
    fields = [
        "name", "uuid", "memory.total", "memory.used", "memory.free",
        "driver_version", "temperature.gpu", "power.draw", "power.limit",
        "clocks.current.sm", "clocks.max.sm",
    ]
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
        line = (out.stdout or "").strip().splitlines()[0]
        result = dict(zip(fields, [v.strip() for v in line.split(",")]))
        try:
            result["_full_nvidia_smi"] = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=30
            ).stdout
        except Exception:
            pass
        return result
    except Exception as exc:
        return {"error": repr(exc)}


def torch_runtime_info() -> Dict[str, Any]:
    try:
        code = "import torch; print(torch.__version__); print(torch.version.cuda)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
        lines = (out.stdout or "").strip().splitlines()
        return {
            "torch": lines[0] if lines else None,
            "cuda_runtime": lines[1] if len(lines) > 1 else None,
        }
    except Exception as exc:
        return {"error": repr(exc)}


def parse_server_log(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return info
    patterns = {
        "kv_cache_tokens": r"GPU KV cache size:\s*([\d,]+)\s*tokens",
        "kv_cache_memory_gib": r"Available KV cache memory:\s*([\d.]+)\s*GiB",
        "max_concurrency": r"Maximum concurrency for\s*[\d,]+\s*tokens per request:\s*([\d.]+)x",
        "block_size": r"(?:attention block size to|block_size[=:\s]+)\s*(\d+)",
        "v1_engine": r"Using V1 engine|V1 engine",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            info[key] = m.group(1) if m.groups() else True
    for key in ("kv_cache_tokens", "max_concurrency"):
        if isinstance(info.get(key), str):
            try:
                info[key] = float(info[key].replace(",", ""))
            except ValueError:
                pass
    if isinstance(info.get("block_size"), str):
        try:
            info["block_size"] = int(info["block_size"])
        except ValueError:
            pass
    lines = text.splitlines()
    info["log_lines"] = [
        l for l in lines
        if re.search(r"KV cache|block size|Maximum concurrency|vLLM API server version|Using V1|prefix cach", l, re.IGNORECASE)
    ][:80]
    info["preempt_mentions"] = len(re.findall(r"preempt", text, re.IGNORECASE))
    info["log_bytes"] = len(text)
    return info


# =============================================================================
# HTTP request execution
# =============================================================================

def chat_payload(
    model: str,
    messages: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    max_tokens: int,
    stream: bool,
) -> Dict[str, Any]:
    b = cfg["benchmark"]
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(b["temperature"]),
        "max_tokens": int(max_tokens),
        "stream": stream,
        "seed": int(b["seed"]),
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def extract_usage(usage: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    usage = usage or {}
    prompt = safe_float(usage.get("prompt_tokens"))
    completion = safe_float(usage.get("completion_tokens"))
    details = usage.get("prompt_tokens_details") or {}
    cached = safe_float(details.get("cached_tokens")) if isinstance(details, dict) else None
    cached_present = cached is not None
    cached_value = cached if cached is not None else 0.0
    uncached = (prompt - cached_value) if prompt is not None else None
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cached_tokens": cached,
        "cached_tokens_present": cached_present,
        "uncached_tokens": uncached,
    }


def prime_chat(
    manager: ServerManager,
    cfg: Dict[str, Any],
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    model = manager.s["served_model_name"]
    payload = chat_payload(model, messages, cfg, int(cfg["benchmark"]["prime_max_tokens"]), stream=False)
    t0 = time.perf_counter()
    r = manager.client.post(f"{manager.base_url}/v1/chat/completions", json=payload)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    data = r.json()
    usage = extract_usage(data.get("usage"))
    choice = (data.get("choices") or [{}])[0]
    text = ((choice.get("message") or {}).get("content") or "")
    return {
        "latency_ms": elapsed_ms,
        "usage": usage,
        "response_text": text,
        "finish_reason": choice.get("finish_reason"),
    }


def stream_chat(
    manager: ServerManager,
    cfg: Dict[str, Any],
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    model = manager.s["served_model_name"]
    payload = chat_payload(model, messages, cfg, int(cfg["benchmark"]["max_tokens"]), stream=True)
    t0 = time.perf_counter()
    first_content: Optional[float] = None
    last_any: Optional[float] = None
    chunk_times: List[float] = []
    parts: List[str] = []
    usage_raw: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    with manager.client.stream(
        "POST", f"{manager.base_url}/v1/chat/completions", json=payload
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            if not data:
                continue
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("usage"):
                usage_raw = obj["usage"]
            for choice in obj.get("choices") or []:
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    now = time.perf_counter()
                    if first_content is None:
                        first_content = now
                    chunk_times.append(now)
                    last_any = now
                    parts.append(piece)
    end = time.perf_counter()
    usage = extract_usage(usage_raw)
    ttft_ms = (first_content - t0) * 1000.0 if first_content is not None else None
    e2e_ms = (end - t0) * 1000.0
    if len(chunk_times) > 1:
        itl_ms = float(np.mean(np.diff(chunk_times)) * 1000.0)
    else:
        itl_ms = None
    return {
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "itl_ms": itl_ms,
        "stream_chunks": len(chunk_times),
        "usage": usage,
        "response_text": "".join(parts),
        "finish_reason": finish_reason,
    }


def value_regex(value: str, kind: str) -> re.Pattern:
    if kind == "int":
        return re.compile(r"(?<![\d.])" + re.escape(value) + r"(?![\d.])")
    return re.compile(re.escape(value), re.IGNORECASE)


def grade_answer(text: str, scenario: Scenario) -> Dict[str, Any]:
    normalized = re.sub(r"(?<=\d),(?=\d)", "", text or "")
    normalized = re.sub(r"[`*_]", "", normalized)
    final_match = re.search(r"FINAL\s*:\s*(.+)", normalized, re.IGNORECASE)
    answer = final_match.group(1).strip().splitlines()[0].strip() if final_match else normalized.strip()
    gt_re = value_regex(scenario.new, scenario.kind)
    stale_re = value_regex(scenario.stale, scenario.kind)
    gt_in_answer = bool(gt_re.search(answer))
    stale_in_answer = bool(stale_re.search(answer))
    stale_in_full = bool(stale_re.search(normalized))
    return {
        "format_ok": final_match is not None,
        "answer_line": answer[:300],
        "gt_in_answer": gt_in_answer,
        "stale_in_answer": stale_in_answer,
        "stale_in_full_text": stale_in_full,
        "evidence_accurate": bool(gt_in_answer and not stale_in_answer),
    }

# =============================================================================
# Trial execution
# =============================================================================

def json_line(record: Dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, default=str)


def execute_trial(
    manager: ServerManager,
    cfg: Dict[str, Any],
    sample: SamplePrompts,
    arm: str,
    phase: str,
    raw_fh: Any,
) -> Dict[str, Any]:
    ap = sample.arms[arm]
    scenario = sample.scenario
    retries = int(cfg["benchmark"]["request_retries"])
    prime_settle = float(cfg["benchmark"]["prime_settle_s"])
    metrics_settle = float(cfg["benchmark"]["metrics_settle_s"])
    last_error: Optional[str] = None

    for attempt in range(1, retries + 2):
        try:
            prime = prime_chat(manager, cfg, ap.prime_messages)
            prime_record = {
                "record_type": "prime",
                "phase": phase,
                "sample_id": sample.sample_id,
                "scenario_id": scenario.sid,
                "rep": sample.rep,
                "arm": arm,
                "salt": ap.salt,
                "attempt": attempt,
                "started_at": utc_now(),
                "prompt_tokens_tokenizer": ap.prime_tokens,
                "prompt_tokens": prime["usage"]["prompt_tokens"],
                "cached_tokens_usage": prime["usage"]["cached_tokens"],
                "prime_latency_ms": prime["latency_ms"],
                "response_text": prime["response_text"][:500],
                "success": True,
            }
            raw_fh.write(json_line(prime_record) + "\n")
            raw_fh.flush()

            time.sleep(prime_settle)
            metrics_before = manager.metrics_snapshot()
            result = stream_chat(manager, cfg, ap.measured_messages)
            time.sleep(metrics_settle)
            metrics_after = manager.metrics_snapshot()

            hits_b, queries_b = prefix_metrics(metrics_before)
            hits_a, queries_a = prefix_metrics(metrics_after)
            cached_metrics = (hits_a - hits_b) if (hits_a is not None and hits_b is not None) else None
            queries_delta = (queries_a - queries_b) if (queries_a is not None and queries_b is not None) else None
            uncached_metrics = (
                (queries_delta - cached_metrics)
                if (queries_delta is not None and cached_metrics is not None)
                else None
            )

            usage = result["usage"]
            usage_prompt = usage["prompt_tokens"]
            usage_cached = usage["cached_tokens"]
            usage_uncached = usage["uncached_tokens"]
            tokenizer_delta = (
                usage_prompt - ap.total_tokens if usage_prompt is not None else None
            )
            grade = grade_answer(result["response_text"], scenario)

            record = {
                "record_type": "measured",
                "phase": phase,
                "sample_id": sample.sample_id,
                "scenario_id": scenario.sid,
                "domain": scenario.domain,
                "rep": sample.rep,
                "arm": arm,
                "salt": ap.salt,
                "attempt": attempt,
                "started_at": utc_now(),
                "prompt_tokens_tokenizer": ap.total_tokens,
                "prompt_tokens": usage_prompt,
                "tokenizer_vs_usage_delta": tokenizer_delta,
                "cached_tokens_usage": usage_cached,
                "cached_tokens_present": usage["cached_tokens_present"],
                "uncached_tokens_usage": usage_uncached,
                "cached_tokens_metrics": cached_metrics,
                "metrics_queries_delta": queries_delta,
                "uncached_tokens_metrics": uncached_metrics,
                "ttft_ms": result["ttft_ms"],
                "e2e_ms": result["e2e_ms"],
                "itl_ms": result["itl_ms"],
                "completion_tokens": usage["completion_tokens"],
                "stream_chunks": result["stream_chunks"],
                "response_text": result["response_text"],
                "finish_reason": result["finish_reason"],
                "grade": grade,
                "construction": sample.checks,
                "prime": {
                    "latency_ms": prime["latency_ms"],
                    "prompt_tokens": prime["usage"]["prompt_tokens"],
                    "cached_tokens": prime["usage"]["cached_tokens"],
                    "prompt_tokens_tokenizer": ap.prime_tokens,
                },
                "prefix_expected": {
                    "field": scenario.field_name,
                    "stale": scenario.stale,
                    "new": scenario.new,
                },
                "success": True,
            }
            raw_fh.write(json_line(record) + "\n")
            raw_fh.flush()
            return record
        except Exception as exc:
            last_error = repr(exc)
            print(f"[trial] {sample.sample_id} {arm} attempt {attempt} failed: {last_error}")
            time.sleep(1.5)

    record = {
        "record_type": "measured",
        "phase": phase,
        "sample_id": sample.sample_id,
        "scenario_id": scenario.sid,
        "rep": sample.rep,
        "arm": arm,
        "salt": ap.salt,
        "started_at": utc_now(),
        "prompt_tokens_tokenizer": ap.total_tokens,
        "ttft_ms": None,
        "e2e_ms": None,
        "success": False,
        "error": last_error,
        "construction": sample.checks,
    }
    raw_fh.write(json_line(record) + "\n")
    raw_fh.flush()
    return record


def run_benchmark(
    manager: ServerManager,
    cfg: Dict[str, Any],
    root: Path,
    scenarios: List[Scenario],
    factory: PromptFactory,
) -> Path:
    b = cfg["benchmark"]
    if int(b["concurrency"]) != 1:
        raise ValueError("This protocol requires concurrency = 1.")

    raw_path = ensure_parent(cfg["outputs"]["raw_jsonl"], root)
    with open(raw_path, "w", encoding="utf-8") as raw_fh:
        print("[run] starting warmup")
        warmup_scenario = scenarios[0]
        warmup_reps = list(range(int(b["warmup_trials"])))
        random.Random(int(b["seed"]) + 17).shuffle(warmup_reps)
        for rep in warmup_reps:
            prompts = factory.build_sample(warmup_scenario, 0, rep, "warmup")
            order = list(ARMS)
            random.Random(int(b["seed"]) * 1009 + rep * 37).shuffle(order)
            for arm in order:
                rec = execute_trial(manager, cfg, prompts, arm, "warmup", raw_fh)
                print(f"[warmup] {prompts.sample_id} {arm} success={rec.get('success')}")
        print("[run] warmup complete")

        measured: List[Tuple[int, int]] = [
            (si, rep)
            for si in range(len(scenarios))
            for rep in range(int(b["trials_per_scenario"]))
        ]
        random.Random(int(b["seed"]) + 23).shuffle(measured)

        total = len(measured)
        done = 0
        for si, rep in measured:
            scenario = scenarios[si]
            prompts = factory.build_sample(scenario, si, rep, "measured")
            checks = prompts.checks
            if not checks["parity_within_tolerance"]:
                print(
                    f"[warn] {prompts.sample_id} parity diff {checks['parity_abs_diff']} "
                    f"(A={checks['arm_a_total_tokens']} B={checks['arm_b_total_tokens']})"
                )
            if not checks["shorter_in_band"]:
                print(f"[warn] {prompts.sample_id} shorter ratio {checks['shorter_ratio']:.3f}")
            order = list(ARMS)
            random.Random(int(b["seed"]) * 1009 + si * 131 + rep * 37).shuffle(order)
            for arm in order:
                rec = execute_trial(manager, cfg, prompts, arm, "measured", raw_fh)
                if rec.get("success"):
                    print(
                        f"[{done + 1}/{total}] {prompts.sample_id} {arm:22s} "
                        f"total={rec.get('prompt_tokens')} cached={rec.get('cached_tokens_usage')} "
                        f"uncached={rec.get('uncached_tokens_usage')} "
                        f"ttft={rec.get('ttft_ms', 0):.1f}ms acc={rec.get('grade', {}).get('evidence_accurate')}"
                    )
                else:
                    print(f"[{done + 1}/{total}] {prompts.sample_id} {arm:22s} FAILED {rec.get('error')}")
            done += 1
            raw_fh.flush()

    print(f"[run] raw records written to {raw_path}")
    return raw_path


# =============================================================================
# Analysis
# =============================================================================

def load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def measured_records(records: List[Dict[str, Any]], phase: str = "measured") -> List[Dict[str, Any]]:
    return [
        r for r in records
        if r.get("record_type") == "measured" and r.get("phase") == phase and r.get("success")
    ]


def get_value(record: Dict[str, Any], field: str) -> Optional[float]:
    return safe_float(record.get(field))


def summarize_arms(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for arm in ARMS:
        subset = [r for r in records if r.get("arm") == arm]
        arm_stats: Dict[str, Any] = {"n_requests": len(subset), "metrics": {}}
        for metric_name, field in METRIC_FIELDS.items():
            arm_stats["metrics"][metric_name] = stats_dict(
                [get_value(r, field) for r in subset]
            )
        correct = [1.0 if (r.get("grade") or {}).get("evidence_accurate") else 0.0 for r in subset]
        arm_stats["accuracy"] = {
            "n": len(correct),
            "correct": int(sum(correct)),
            "accuracy_pct": float(100.0 * np.mean(correct)) if correct else None,
        }
        arm_stats["cache_hit_fraction_usage"] = (
            float(np.mean([
                (get_value(r, "cached_tokens_usage") or 0.0) / max(1.0, get_value(r, "prompt_tokens") or 1.0)
                for r in subset
            ]))
            if subset else None
        )
        arm_stats["cache_hit_fraction_metrics"] = (
            float(np.mean([
                (get_value(r, "cached_tokens_metrics") or 0.0)
                / max(1.0, (get_value(r, "metrics_queries_delta") or 1.0))
                for r in subset
            ]))
            if subset else None
        )
        out[arm] = arm_stats
    return out


def paired_comparisons(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_sample: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in records:
        by_sample[r["sample_id"]][r["arm"]] = r

    metrics = {
        "total_input_tokens": "prompt_tokens",
        "cached_input_tokens_usage": "cached_tokens_usage",
        "uncached_input_tokens_usage": "uncached_tokens_usage",
        "uncached_input_tokens_metrics": "uncached_tokens_metrics",
        "ttft_ms": "ttft_ms",
        "e2e_ms": "e2e_ms",
    }
    out: Dict[str, Any] = {}
    for comparison_arm in [ARM_EQUAL, ARM_SHORTER]:
        key = f"{comparison_arm}_vs_{ARM_APPEND}"
        comparison: Dict[str, Any] = {"pairs": 0, "metrics": {}}
        for metric_name, field in metrics.items():
            deltas: List[float] = []
            ratios: List[float] = []
            for sample_id, arms in by_sample.items():
                if ARM_APPEND not in arms or comparison_arm not in arms:
                    continue
                a = get_value(arms[ARM_APPEND], field)
                c = get_value(arms[comparison_arm], field)
                if a is None or c is None:
                    continue
                deltas.append(c - a)
                if a != 0:
                    ratios.append(c / a)
            comparison["pairs"] = max(comparison["pairs"], len(deltas))
            comparison["metrics"][metric_name] = {
                "delta": stats_dict(deltas),
                "delta_mean_95ci": bootstrap_mean_ci(deltas),
                "ratio": stats_dict(ratios),
            }
        out[key] = comparison

    # Accuracy agreement across arms (paired).
    accuracy_pairs = []
    for arms in by_sample.values():
        if all(a in arms for a in ARMS):
            values = [bool((arms[a].get("grade") or {}).get("evidence_accurate")) for a in ARMS]
            accuracy_pairs.append(values)
    if accuracy_pairs:
        arr = np.asarray(accuracy_pairs, dtype=float)
        out["accuracy_pairwise"] = {
            "pairs": len(accuracy_pairs),
            "all_three_correct_pct": float(100.0 * np.mean(arr.all(axis=1))),
            "per_arm_pct": {arm: float(100.0 * np.mean(arr[:, i])) for i, arm in enumerate(ARMS)},
        }
    return out


def derive_metrics(records: List[Dict[str, Any]], arm_stats: Dict[str, Any]) -> Dict[str, Any]:
    xs: List[float] = []
    ys: List[float] = []
    for r in records:
        u = get_value(r, "uncached_tokens_usage")
        t = get_value(r, "ttft_ms")
        if u is not None and t is not None:
            xs.append(u / 1000.0)
            ys.append(t)
    if len(xs) >= 2:
        slope, intercept = np.polyfit(np.asarray(xs), np.asarray(ys), 1)
        pred = slope * np.asarray(xs) + intercept
        ss_res = float(np.sum((np.asarray(ys) - pred) ** 2))
        ss_tot = float(np.sum((np.asarray(ys) - np.mean(ys)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
        n_uncached = len(xs)
    else:
        slope = intercept = r2 = None
        n_uncached = len(xs)

    def mean_metric(arm: str, metric: str) -> Optional[float]:
        return (arm_stats.get(arm, {}).get("metrics", {}).get(metric, {}) or {}).get("mean")

    t_a = mean_metric(ARM_APPEND, "total_input_tokens")
    t_b = mean_metric(ARM_EQUAL, "total_input_tokens")
    t_c = mean_metric(ARM_SHORTER, "total_input_tokens")
    u_a = mean_metric(ARM_APPEND, "uncached_input_tokens_usage")
    u_b = mean_metric(ARM_EQUAL, "uncached_input_tokens_usage")
    u_c = mean_metric(ARM_SHORTER, "uncached_input_tokens_usage")
    p_a = mean_metric(ARM_APPEND, "cached_input_tokens_usage")
    p_b = mean_metric(ARM_EQUAL, "cached_input_tokens_usage")
    p_c = mean_metric(ARM_SHORTER, "cached_input_tokens_usage")
    ttft_a = mean_metric(ARM_APPEND, "ttft_ms")
    ttft_b = mean_metric(ARM_EQUAL, "ttft_ms")
    ttft_c = mean_metric(ARM_SHORTER, "ttft_ms")

    cached_after_rewrite = None
    if p_b is not None and p_c is not None:
        cached_after_rewrite = float((p_b + p_c) / 2.0)
    break_even_ratio = None
    break_even_tokens = None
    if None not in (u_a, t_a) and cached_after_rewrite is not None and t_a:
        break_even_tokens = float(u_a + cached_after_rewrite)
        break_even_ratio = float(break_even_tokens / t_a)

    paradox = {}
    if None not in (t_a, t_c, ttft_a, ttft_c) and t_a:
        paradox = {
            "total_token_reduction_pct_C_vs_A": float(100.0 * (1.0 - t_c / t_a)),
            "ttft_change_pct_C_vs_A": float(100.0 * (ttft_c / ttft_a - 1.0)),
            "uncached_token_ratio_C_vs_A": float(u_c / u_a) if u_a else None,
            "cached_token_ratio_C_vs_A": float(p_c / p_a) if p_a else None,
        }

    cache_invalidation = {}
    if None not in (ttft_a, ttft_b, u_a, u_b) and slope is not None:
        delta_ttft = ttft_b - ttft_a
        delta_uncached = u_b - u_a
        cache_invalidation = {
            "ttft_delta_B_minus_A_ms": float(delta_ttft),
            "uncached_delta_B_minus_A": float(delta_uncached),
            "predicted_from_slope_ms": float(slope * delta_uncached / 1000.0),
            "residual_ms": float(delta_ttft - slope * delta_uncached / 1000.0),
        }

    return {
        "regression_ttft_vs_uncached": {
            "n": n_uncached,
            "slope_ms_per_1k_uncached_tokens": float(slope) if slope is not None else None,
            "intercept_ms": float(intercept) if intercept is not None else None,
            "r2": float(r2) if r2 is not None else None,
        },
        "means": {
            "total_tokens_A": t_a,
            "total_tokens_B": t_b,
            "total_tokens_C": t_c,
            "uncached_tokens_A": u_a,
            "uncached_tokens_B": u_b,
            "uncached_tokens_C": u_c,
            "cached_tokens_A": p_a,
            "cached_tokens_B": p_b,
            "cached_tokens_C": p_c,
            "ttft_ms_A": ttft_a,
            "ttft_ms_B": ttft_b,
            "ttft_ms_C": ttft_c,
        },
        "cached_prefix_after_rewrite_tokens": cached_after_rewrite,
        "break_even": {
            "ratio_of_A_total": break_even_ratio,
            "absolute_total_tokens": break_even_tokens,
            "formula": "r* = (uncached_A + cached_after_rewrite) / total_A",
        },
        "shorter_prompt_paradox": paradox,
        "cache_invalidation_cost": cache_invalidation,
    }

def run_checks(
    records: List[Dict[str, Any]],
    arm_stats: Dict[str, Any],
    derived: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    measured = measured_records(records)
    by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in measured:
        by_sample[r["sample_id"]].append(r)

    parity_diffs = []
    shorter_ratios = []
    prime_a_tokens = []
    for sample_id, recs in by_sample.items():
        if recs:
            c = recs[0].get("construction") or {}
            if "parity_abs_diff" in c:
                parity_diffs.append(float(c["parity_abs_diff"]))
            if "shorter_ratio" in c:
                shorter_ratios.append(float(c["shorter_ratio"]))
            if "prime_tokens_a" in c:
                prime_a_tokens.append(float(c["prime_tokens_a"]))

    usage_metric_pairs = [
        (get_value(r, "cached_tokens_usage"), get_value(r, "cached_tokens_metrics"))
        for r in measured
        if get_value(r, "cached_tokens_usage") is not None
        and get_value(r, "cached_tokens_metrics") is not None
    ]
    mean_abs_cached_mismatch = (
        float(np.mean([abs(a - b) for a, b in usage_metric_pairs]))
        if usage_metric_pairs else None
    )
    cached_corr = None
    if len(usage_metric_pairs) >= 2:
        aa = np.asarray([p[0] for p in usage_metric_pairs], dtype=float)
        bb = np.asarray([p[1] for p in usage_metric_pairs], dtype=float)
        if aa.std() > 0 and bb.std() > 0:
            cached_corr = float(np.corrcoef(aa, bb)[0, 1])

    tokenizer_usage_deltas = [
        abs(get_value(r, "tokenizer_vs_usage_delta"))
        for r in measured
        if get_value(r, "tokenizer_vs_usage_delta") is not None
    ]

    accuracies = {
        arm: (arm_stats[arm]["accuracy"]["accuracy_pct"] if arm in arm_stats else None)
        for arm in ARMS
    }
    hit_a = arm_stats.get(ARM_APPEND, {}).get("cache_hit_fraction_usage")
    hit_b = arm_stats.get(ARM_EQUAL, {}).get("cache_hit_fraction_usage")
    hit_c = arm_stats.get(ARM_SHORTER, {}).get("cache_hit_fraction_usage")
    paradox = derived.get("shorter_prompt_paradox", {})

    return {
        "measured_requests": len(measured),
        "expected_measured_requests": int(cfg["benchmark"]["num_scenarios"])
        * int(cfg["benchmark"]["trials_per_scenario"])
        * len(ARMS),
        "samples": len(by_sample),
        "parity_max_abs_diff": max(parity_diffs) if parity_diffs else None,
        "parity_mean_abs_diff": float(np.mean(parity_diffs)) if parity_diffs else None,
        "shorter_ratio_min": min(shorter_ratios) if shorter_ratios else None,
        "shorter_ratio_max": max(shorter_ratios) if shorter_ratios else None,
        "shorter_ratio_mean": float(np.mean(shorter_ratios)) if shorter_ratios else None,
        "prime_tokens_a_mean": float(np.mean(prime_a_tokens)) if prime_a_tokens else None,
        "cache_hit_fraction_usage_A": hit_a,
        "cache_hit_fraction_usage_B": hit_b,
        "cache_hit_fraction_usage_C": hit_c,
        "append_only_high_cache_hit": bool(hit_a is not None and hit_a > 0.5),
        "rewrite_low_cache_hit": bool(
            hit_b is not None and hit_c is not None and hit_b < 0.2 and hit_c < 0.2
        ),
        "cached_usage_vs_metrics_mean_abs_diff": mean_abs_cached_mismatch,
        "cached_usage_vs_metrics_correlation": cached_corr,
        "tokenizer_vs_usage_mean_abs_delta": float(np.mean(tokenizer_usage_deltas))
        if tokenizer_usage_deltas else None,
        "accuracy_pct": accuracies,
        "all_arms_evidence_accurate_100pct": all(
            (v is not None and abs(v - 100.0) < 1e-9) for v in accuracies.values()
        ),
        "shorter_prompt_C_has_more_uncached_than_A": (
            paradox.get("uncached_token_ratio_C_vs_A") is not None
            and paradox["uncached_token_ratio_C_vs_A"] > 1.0
        ),
        "shorter_prompt_C_ttft_slower_than_A": (
            paradox.get("ttft_change_pct_C_vs_A") is not None
            and paradox["ttft_change_pct_C_vs_A"] > 0
        ),
        "parity_enforced": (
            max(parity_diffs) <= float(cfg["benchmark"]["token_parity_tolerance"])
            if parity_diffs else False
        ),
        "shorter_ratio_in_band": (
            min(shorter_ratios) >= float(cfg["benchmark"]["shorter_prompt_min_ratio"])
            and max(shorter_ratios) <= float(cfg["benchmark"]["shorter_prompt_max_ratio"])
            if shorter_ratios else False
        ),
    }


def write_summary(
    cfg: Dict[str, Any],
    root: Path,
    records: List[Dict[str, Any]],
    telemetry: Dict[str, Any],
) -> Dict[str, Any]:
    measured = measured_records(records)
    arm_stats = summarize_arms(measured)
    paired = paired_comparisons(measured)
    derived = derive_metrics(measured, arm_stats)
    checks = run_checks(records, arm_stats, derived, cfg)

    run_metrics_start = telemetry.get("run_metrics_start") or {}
    run_metrics_end = telemetry.get("run_metrics_end") or {}
    preempt_delta = None
    if preemptions(run_metrics_start) is not None and preemptions(run_metrics_end) is not None:
        preempt_delta = preemptions(run_metrics_end) - preemptions(run_metrics_start)

    summary = {
        "generated_at": utc_now(),
        "config": cfg,
        "environment": {
            "gpu": telemetry.get("gpu_start"),
            "versions": telemetry.get("versions"),
            "torch_runtime": telemetry.get("torch_runtime"),
            "serve_command": telemetry.get("serve_command"),
            "server_reused": telemetry.get("server_reused"),
            "server": telemetry.get("server"),
        },
        "run": {
            "measured_requests": len(measured),
            "warmup_requests": len([
                r for r in records
                if r.get("record_type") == "measured" and r.get("phase") == "warmup"
            ]),
            "failed_requests": len([
                r for r in records
                if r.get("record_type") == "measured" and not r.get("success")
            ]),
            "preemptions_delta": preempt_delta,
            "gpu_end": telemetry.get("gpu_end"),
            "run_metrics_start": run_metrics_start,
            "run_metrics_end": run_metrics_end,
        },
        "arms": arm_stats,
        "paired": paired,
        "derived": derived,
        "checks": checks,
    }

    json_path = ensure_parent(cfg["outputs"]["summary_json"], root)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"[analysis] wrote {json_path}")

    csv_path = ensure_parent(cfg["outputs"]["summary_csv"], root)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["arm", "metric", "n", "mean", "median_p50", "p95", "std", "min", "max"],
        )
        writer.writeheader()
        for arm in ARMS:
            for metric_name, st in arm_stats[arm]["metrics"].items():
                writer.writerow({
                    "arm": arm,
                    "metric": metric_name,
                    "n": st["n"],
                    "mean": st["mean"],
                    "median_p50": st["median"],
                    "p95": st["p95"],
                    "std": st["std"],
                    "min": st["min"],
                    "max": st["max"],
                })
            acc = arm_stats[arm]["accuracy"]
            writer.writerow({
                "arm": arm,
                "metric": "evidence_accuracy_pct",
                "n": acc["n"],
                "mean": acc["accuracy_pct"],
                "median_p50": acc["accuracy_pct"],
                "p95": acc["accuracy_pct"],
                "std": 0.0,
                "min": acc["accuracy_pct"],
                "max": acc["accuracy_pct"],
            })
    print(f"[analysis] wrote {csv_path}")
    return summary


def _plot_style() -> None:
    if plt is None:
        return
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "legend.frameon": False,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
    })


def _jitter(n: int, width: float = 0.13, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-width, width, size=n)


def make_charts(
    cfg: Dict[str, Any],
    root: Path,
    records: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> List[str]:
    if plt is None:
        print("[charts] matplotlib unavailable; skipping")
        return []
    _plot_style()
    measured = measured_records(records)
    if not measured:
        print("[charts] no measured records; skipping")
        return []

    produced: List[str] = []
    charts = cfg["outputs"]["charts"]

    # ---- ttft_comparison.png: TTFT and uncached-token distributions ----------
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for ax, field, title, ylabel in [
        (axes[0], "ttft_ms", "Time to first token", "TTFT (ms)"),
        (axes[1], "uncached_tokens_usage", "Newly prefetched input", "Uncached input tokens"),
    ]:
        data = []
        for arm in ARMS:
            vals = [get_value(r, field) for r in measured if r.get("arm") == arm]
            data.append([v for v in vals if v is not None])
        bp = ax.boxplot(
            data, widths=0.5, patch_artist=True, showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.5},
        )
        for patch, arm in zip(bp["boxes"], ARMS):
            patch.set_facecolor(ARM_COLOR[arm])
            patch.set_alpha(0.22)
            patch.set_edgecolor(ARM_COLOR[arm])
        for i, arm in enumerate(ARMS):
            vals = data[i]
            if not vals:
                continue
            ax.scatter(
                np.full(len(vals), i + 1) + _jitter(len(vals)),
                vals, s=14, alpha=0.55, color=ARM_COLOR[arm], edgecolors="none", zorder=3,
            )
            mean = float(np.mean(vals))
            ax.scatter([i + 1], [mean], marker="D", s=42, color=ARM_COLOR[arm],
                       edgecolors="white", linewidths=0.8, zorder=4)
            ax.annotate(
                f"µ {mean:,.1f}",
                (i + 1, mean), xytext=(8, 8), textcoords="offset points",
                fontsize=8.5, color=ARM_COLOR[arm], fontweight="bold",
            )
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([ARM_LABEL[a].replace(" · ", "\n") for a in ARMS], fontsize=8.2)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    fig.suptitle("Paired prefix-cache experiment — Qwen2.5-14B-Instruct / vLLM / RTX PRO 6000 Blackwell",
                 fontsize=11.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = ensure_parent(charts["ttft_comparison"], root)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    produced.append(str(out))
    print(f"[charts] wrote {out}")

    # ---- uncached_tokens_vs_ttft.png ----------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for arm in ARMS:
        xs, ys = [], []
        for r in measured:
            if r.get("arm") != arm:
                continue
            u = get_value(r, "uncached_tokens_usage")
            t = get_value(r, "ttft_ms")
            if u is not None and t is not None:
                xs.append(u)
                ys.append(t)
        ax.scatter(xs, ys, s=30, alpha=0.55, color=ARM_COLOR[arm],
                   edgecolors="white", linewidths=0.4, label=ARM_LABEL[arm])
        if xs:
            ax.scatter([np.mean(xs)], [np.mean(ys)], marker="D", s=75,
                       color=ARM_COLOR[arm], edgecolors="black", linewidths=0.7, zorder=5)
    valid = [
        (get_value(r, "uncached_tokens_usage"), get_value(r, "ttft_ms"))
        for r in measured
        if get_value(r, "uncached_tokens_usage") is not None and get_value(r, "ttft_ms") is not None
    ]
    if len(valid) >= 2:
        x = np.asarray([v[0] for v in valid], dtype=float) / 1000.0
        y = np.asarray([v[1] for v in valid], dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 100)
        ax.plot(grid, slope * grid + intercept, color="#222222", linewidth=1.6,
                linestyle="--", label=f"OLS: {slope:.1f} ms / 1k uncached + {intercept:.1f} ms")
        pred = slope * x + intercept
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        ax.text(
            0.02, 0.98,
            f"n = {len(valid)}\nslope = {slope:.1f} ms / 1k uncached tokens\n"
            f"intercept = {intercept:.1f} ms\nR² = {r2:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.7,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#cccccc", alpha=0.9),
        )
    ax.set_xlabel("Uncached (newly prefetched) input tokens (thousands)")
    ax.set_ylabel("Time to first token (ms)")
    ax.set_title("TTFT is explained by uncached prefill volume, not total prompt length")
    ax.legend(loc="lower right", fontsize=8.2)
    fig.tight_layout()
    out = ensure_parent(charts["uncached_tokens_vs_ttft"], root)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    produced.append(str(out))
    print(f"[charts] wrote {out}")

    # ---- shorter_prompt_paradox.png -----------------------------------------
    means = summary["derived"]["means"]
    panels = [
        ("total_tokens_A", "total_tokens_C", "Total input tokens", "tokens", "{:,.0f}"),
        ("uncached_tokens_A", "uncached_tokens_C", "Uncached input tokens", "tokens", "{:,.0f}"),
        ("ttft_ms_A", "ttft_ms_C", "Time to first token", "ms", "{:,.1f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0))
    for ax, (ka, kc, title, unit, fmt) in zip(axes, panels):
        va = means.get(ka)
        vc = means.get(kc)
        if va is None or vc is None:
            ax.set_visible(False)
            continue
        bars = ax.bar(
            [0, 1], [va, vc], width=0.58,
            color=[ARM_COLOR[ARM_APPEND], ARM_COLOR[ARM_SHORTER]],
            alpha=0.82, edgecolor="white",
        )
        for bar, val in zip(bars, [va, vc]):
            ax.annotate(
                fmt.format(val), (bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold",
            )
        delta_pct = 100.0 * (vc / va - 1.0) if va else None
        if delta_pct is not None:
            ax.annotate(
                f"{delta_pct:+.1f}%", (0.5, 0.93 * max(va, vc)),
                xytext=(0, 0), textcoords="offset points",
                ha="center", fontsize=11, fontweight="bold",
                color="#2f7d32" if delta_pct < 0 else "#b3261e",
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["A\nappend", "C\nshorter"], fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(unit)
    paradox = summary["derived"].get("shorter_prompt_paradox", {})
    if paradox:
        fig.suptitle(
            "The shorter-prompt paradox: "
            f"{paradox.get('total_token_reduction_pct_C_vs_A', 0):.1f}% fewer total tokens, "
            f"{paradox.get('ttft_change_pct_C_vs_A', 0):+.1f}% TTFT",
            fontsize=11.5, fontweight="bold", y=1.03,
        )
    fig.tight_layout()
    out = ensure_parent(charts["shorter_prompt_paradox"], root)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    produced.append(str(out))
    print(f"[charts] wrote {out}")

    return produced


def analyze_results(cfg: Dict[str, Any], root: Path) -> Dict[str, Any]:
    raw_path = resolve_path(cfg["outputs"]["raw_jsonl"], root)
    records = load_records(raw_path)
    telemetry_path = resolve_path(cfg["outputs"]["telemetry_json"], root)
    telemetry: Dict[str, Any] = {}
    if telemetry_path.exists():
        try:
            telemetry = json.loads(telemetry_path.read_text())
        except Exception:
            telemetry = {}
    if not telemetry:
        telemetry = {
            "gpu_start": nvidia_smi_query() if shutil.which("nvidia-smi") else {},
            "versions": {
                "vllm": package_version("vllm"),
                "torch": package_version("torch"),
                "transformers": package_version("transformers"),
            },
            "server": parse_server_log(resolve_path(cfg["outputs"]["server_log"], root)),
        }
    summary = write_summary(cfg, root, records, telemetry)
    make_charts(cfg, root, records, summary)
    return summary

# =============================================================================
# Orchestration
# =============================================================================

def write_serve_config(cfg: Dict[str, Any], root: Path, manager: ServerManager) -> None:
    path = ensure_parent(cfg["outputs"]["serve_config_json"], root)
    payload = {
        "captured_at": utc_now(),
        "serve_command": manager.serve_command,
        "server_reused": not manager.started_by_us,
        "vllm_server": cfg["vllm_server"],
        "versions": {
            "vllm": package_version("vllm"),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
        },
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"[server] wrote {path}")


def build_telemetry(cfg: Dict[str, Any], root: Path, manager: ServerManager) -> Dict[str, Any]:
    telemetry: Dict[str, Any] = {
        "captured_at": utc_now(),
        "gpu_start": nvidia_smi_query(),
        "versions": {
            "vllm": package_version("vllm"),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "httpx": package_version("httpx"),
        },
        "torch_runtime": torch_runtime_info(),
        "serve_command": manager.serve_command,
        "server_reused": not manager.started_by_us,
        "run_metrics_start": manager.metrics_snapshot(),
        "server_log_path": str(resolve_path(cfg["outputs"]["server_log"], root)),
    }
    return telemetry


def run_full(cfg: Dict[str, Any], root: Path, args: argparse.Namespace) -> None:
    if AutoTokenizer is None:
        raise RuntimeError("transformers is required (pip install transformers)")

    if cfg["vllm_server"].get("hf_home"):
        os.environ.setdefault("HF_HOME", str(cfg["vllm_server"]["hf_home"]))

    seed = int(cfg["benchmark"]["seed"])
    random.seed(seed)
    np.random.seed(seed)

    print(f"[init] loading tokenizer: {cfg['vllm_server']['model']}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["vllm_server"]["model"],
        trust_remote_code=bool(cfg["vllm_server"].get("trust_remote_code", False)),
    )
    print("[init] building scenarios")
    scenarios = build_scenarios(tokenizer, cfg)
    for sc in scenarios:
        print(f"[scenario] {sc.sid:24s} tool_tokens={sc.tool_text_tokens} field={sc.field_name}")
    factory = PromptFactory(tokenizer, cfg)
    manager = ServerManager(cfg, root)
    telemetry: Dict[str, Any] = {}
    try:
        if args.start_server:
            manager.start()
        else:
            if not manager.healthy():
                raise RuntimeError("Server is not healthy and --no-start-server was given.")
            manager.verify_model()
        write_serve_config(cfg, root, manager)
        telemetry = build_telemetry(cfg, root, manager)

        print(f"[bench] config: {cfg['benchmark']['num_scenarios']} scenarios x "
              f"{cfg['benchmark']['trials_per_scenario']} reps x {len(ARMS)} arms")
        raw_path = run_benchmark(manager, cfg, root, scenarios, factory)
        telemetry["gpu_end"] = nvidia_smi_query()
        telemetry["run_metrics_end"] = manager.metrics_snapshot()
        telemetry["server"] = parse_server_log(resolve_path(cfg["outputs"]["server_log"], root))
        if (
            preemptions(telemetry.get("run_metrics_start") or {}) is not None
            and preemptions(telemetry.get("run_metrics_end") or {}) is not None
        ):
            telemetry["preemptions_delta"] = (
                preemptions(telemetry["run_metrics_end"])
                - preemptions(telemetry["run_metrics_start"])
            )
        telemetry["finished_at"] = utc_now()

        telemetry_path = ensure_parent(cfg["outputs"]["telemetry_json"], root)
        with open(telemetry_path, "w", encoding="utf-8") as fh:
            json.dump(telemetry, fh, indent=2, default=str)
        print(f"[server] wrote {telemetry_path}")

        summary = analyze_results(cfg, root)
        print("\n=== HEADLINE ===")
        means = summary["derived"]["means"]
        checks = summary["checks"]
        print(f"A total={means['total_tokens_A']:.0f} cached={means['cached_tokens_A']:.0f} "
              f"uncached={means['uncached_tokens_A']:.0f} ttft={means['ttft_ms_A']:.1f}ms")
        print(f"B total={means['total_tokens_B']:.0f} cached={means['cached_tokens_B']:.0f} "
              f"uncached={means['uncached_tokens_B']:.0f} ttft={means['ttft_ms_B']:.1f}ms")
        print(f"C total={means['total_tokens_C']:.0f} cached={means['cached_tokens_C']:.0f} "
              f"uncached={means['uncached_tokens_C']:.0f} ttft={means['ttft_ms_C']:.1f}ms")
        print(f"accuracy: {checks['accuracy_pct']}")
        print(f"parity max diff: {checks['parity_max_abs_diff']} | "
              f"shorter ratio: {checks['shorter_ratio_mean']:.3f} | "
              f"preemptions delta: {summary['run'].get('preemptions_delta')}")
        print(f"raw: {raw_path}")
    finally:
        if args.keep_server:
            print("[server] leaving vLLM running (--keep-server)")
        else:
            manager.stop()
        manager.close()


def run_dry(cfg: Dict[str, Any], root: Path, args: argparse.Namespace) -> None:
    if AutoTokenizer is None:
        raise RuntimeError("transformers is required for --dry-run")
    if cfg["vllm_server"].get("hf_home"):
        os.environ.setdefault("HF_HOME", str(cfg["vllm_server"]["hf_home"]))
    print(f"[dry-run] loading tokenizer: {cfg['vllm_server']['model']}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["vllm_server"]["model"],
        trust_remote_code=bool(cfg["vllm_server"].get("trust_remote_code", False)),
    )
    scenarios = build_scenarios(tokenizer, cfg)
    factory = PromptFactory(tokenizer, cfg)
    n = max(1, int(args.dry_run_samples))
    all_checks = []
    for i in range(min(n, len(scenarios))):
        sc = scenarios[i]
        prompts = factory.build_sample(sc, i, 0, "measured")
        checks = prompts.checks
        all_checks.append(checks)
        print(f"[dry-run] {sc.sid}")
        for arm in ARMS:
            ap = prompts.arms[arm]
            print(f"  {arm:22s} prime={ap.prime_tokens:5d} total={ap.total_tokens:5d} "
                  f"notes={ap.summary_note_count:3d} pad={ap.summary_pad_units:3d} salt={ap.salt}")
        print(f"  parity_diff={checks['parity_abs_diff']} within={checks['parity_within_tolerance']} "
              f"shorter_ratio={checks['shorter_ratio']:.3f} in_band={checks['shorter_in_band']}")
        print(f"  tool_tokens={sc.tool_text_tokens} field={sc.field_name} "
              f"stale={sc.stale} new={sc.new}")
    if all_checks:
        print("\n[dry-run] summary:")
        print(f"  max parity diff: {max(c['parity_abs_diff'] for c in all_checks)}")
        print(f"  shorter ratio range: {min(c['shorter_ratio'] for c in all_checks):.3f} .. "
              f"{max(c['shorter_ratio'] for c in all_checks):.3f}")
        print(f"  total A range: {min(c['arm_a_total_tokens'] for c in all_checks)} .. "
              f"{max(c['arm_a_total_tokens'] for c in all_checks)}")
        print(f"  prime A range: {min(c['prime_tokens_a'] for c in all_checks)} .. "
              f"{max(c['prime_tokens_a'] for c in all_checks)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="run.yaml", help="Path to run.yaml")
    parser.add_argument("--analyze-only", action="store_true", help="Regenerate summary/charts from raw JSONL")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts and print token checks; no server")
    parser.add_argument("--dry-run-samples", type=int, default=3, help="Samples to build in dry-run")
    parser.add_argument("--no-start-server", dest="start_server", action="store_false",
                        help="Require an already-healthy server instead of starting one")
    parser.add_argument("--keep-server", action="store_true", help="Do not stop a server started by this script")
    parser.set_defaults(start_server=True)
    args = parser.parse_args()

    cfg, root = load_config(args.config)
    (root / cfg["outputs"]["dir"]).mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        summary = analyze_results(cfg, root)
        print(json.dumps(summary["checks"], indent=2))
        return 0
    if args.dry_run:
        run_dry(cfg, root, args)
        return 0
    run_full(cfg, root, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
