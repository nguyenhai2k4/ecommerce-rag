"""
Evaluation & Benchmarking Suite for Nexus Specialized Document Agent
"""
def run_agent_benchmark(*args, **kwargs):
    from src.evaluation.eval_suite import run_agent_benchmark as _run
    return _run(*args, **kwargs)

__all__ = ["run_agent_benchmark"]
