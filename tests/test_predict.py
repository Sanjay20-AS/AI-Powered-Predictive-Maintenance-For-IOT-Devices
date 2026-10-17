import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from predict import recommendation

def test_recommendation_includes_risk_level():
    result = recommendation({"temperature_c": 90, "vibration_mm_s": 8, "pressure_bar": 4, "hours_since_service": 1200}, .85)
    assert any("HIGH MODEL RISK" in x for x in result)
    assert len(result) >= 4
