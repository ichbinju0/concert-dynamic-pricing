import os
import shutil
import pytest
from unittest.mock import patch, MagicMock


def test_full_pipeline_smoke(tmp_path, monkeypatch):
    """End-to-end smoke test with mocked Claude API."""
    monkeypatch.chdir(tmp_path)
    os.makedirs(tmp_path / "results", exist_ok=True)
    shutil.copytree(
        os.path.join(os.path.dirname(__file__), "..", "data"),
        tmp_path / "data",
    )

    mock_claude = MagicMock()
    mock_claude.messages.create.return_value = MagicMock(
        content=[MagicMock(
            text='{"zones":{"A":{"Z1":0.9,"Z2":0.9,"Z3":1},'
                 '"B":{"Z1":0.5,"Z2":0.6,"Z3":0},'
                 '"C":{"Z1":0.2,"Z2":0.3,"Z3":0}}}'
        )]
    )
    with patch("agents.seat_agent.anthropic.Anthropic", return_value=mock_claude):
        from main import run_agent
        result = run_agent({
            "artist":            "aespa",
            "popularity_score":  6,
            "venue":             "KSPO Dome",
            "venue_description": "원형 공연장, A구역 무대 정면, B구역 측면, C구역 2층, 런웨이 있음",
            "total_seats":       15000,
            "official_price":    154000,
            "sale_start_d_day":  60,
        })

    assert "kpi" in result
    assert "report_path" in result
    assert result["kpi"]["revenue_gain_pct"] is not None
    assert os.path.exists(result["report_path"])
