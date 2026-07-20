"""
MCP Server #2 — Stats / Compute server.

Exposes the Analysis Agent's "external world": structured quarterly
metrics plus real numeric compute (trend, % change, simple linear
projection). Transport: stdio.
"""
import json
import statistics
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "metrics.json"

mcp = FastMCP("stats-server")


def _load() -> dict:
    return json.loads(DATA_FILE.read_text())


@mcp.tool()
def get_metric(metric_name: str) -> str:
    """Return the quarterly series for a metric.

    Valid metric_name values: revenue_growth_pct, churn_rate_pct, csat_score,
    support_ticket_volume, eu_revenue_share_pct.
    """
    data = _load()
    if metric_name not in data:
        return f"Unknown metric '{metric_name}'. Available: {', '.join(data.keys())}"
    return json.dumps(data[metric_name])


@mcp.tool()
def compute_trend(metric_name: str) -> str:
    """Compute the quarter-over-quarter percent change and overall trend direction for a metric."""
    data = _load()
    if metric_name not in data:
        return f"Unknown metric '{metric_name}'. Available: {', '.join(data.keys())}"
    series = data[metric_name]
    quarters = sorted(series.keys())
    values = [series[q] for q in quarters]

    changes = []
    for i in range(1, len(values)):
        prev, cur = values[i - 1], values[i]
        pct_change = ((cur - prev) / prev) * 100 if prev else float("inf")
        changes.append(f"{quarters[i-1]}->{quarters[i]}: {pct_change:+.1f}%")

    direction = "increasing" if values[-1] > values[0] else "decreasing"
    slope = (values[-1] - values[0]) / (len(values) - 1)

    return json.dumps({
        "metric": metric_name,
        "values_by_quarter": series,
        "qoq_pct_changes": changes,
        "overall_direction": direction,
        "avg_qoq_slope": round(slope, 3),
        "projected_next_quarter": round(values[-1] + slope, 3),
    })


@mcp.tool()
def summarize_all_metrics() -> str:
    """Return mean and latest value for every tracked metric, in one call."""
    data = _load()
    summary = {}
    for name, series in data.items():
        values = list(series.values())
        summary[name] = {
            "mean": round(statistics.mean(values), 3),
            "latest": values[-1],
        }
    return json.dumps(summary)


if __name__ == "__main__":
    mcp.run(transport="stdio")
