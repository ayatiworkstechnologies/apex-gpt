from datetime import date
from pathlib import Path
from uuid import uuid4

from data.update_prices import parse_todaypricerates, update_prices


def test_todaypricerates_parser_extracts_material_rates():
    html = """
    <h2>Price of Cement & Sand in chennai</h2>
    Cement Bag ₹ 340 ₹ 365 ₹ 395
    River Sand Unit (100 CFT) ₹ 7800 ₹ 8400 ₹ 9050
    M Sand Unit (100 CFT) ₹ 5500 ₹ 5900 ₹ 6400
    Construction Bricks Rate
    Clay Bricks Piece ₹ 11 ₹ 12 ₹ 13
    Gravel CFT ₹ 42 ₹ 46 ₹ 50
    TMT Steel Ton ₹ 56500 ₹ 61000 ₹ 65000
    """

    assert parse_todaypricerates(html) == {
        "cement": 365,
        "sand": 59,
        "brick": 12,
        "aggregate": 46,
        "steel": 61,
    }


def test_todaypricerates_parser_falls_back_to_detail_ranges():
    html = """
    OPC 53 Grade Cement: ₹380 – ₹440 per 50 kg bag
    Fe 500: ₹58,000 – ₹64,000 per metric ton
    Red Clay Bricks: ₹6 – ₹10 per piece
    River Sand: ₹1,300 – ₹1,700 per ton
    20mm Aggregate: ₹1,300 – ₹1,700 per ton
    """

    assert parse_todaypricerates(html) == {
        "cement": 410,
        "sand": 60,
        "brick": 8,
        "aggregate": 45,
        "steel": 61,
    }


def test_todaypricerates_parser_handles_unit_based_ranges():
    html = """
    River Sand: ₹4,500 – ₹6,200 per unit
    M-Sand: ₹3,300 – ₹4,500 per unit
    20mm Aggregate: ₹1,300 – ₹1,700 per ton
    Red Clay Bricks: ₹9 – ₹12 per piece
    Fe 500: ₹59,000 – ₹63,000 per metric ton
    OPC 53 Grade Cement: ₹390 – ₹440 per 50 kg bag
    """

    assert parse_todaypricerates(html) == {
        "cement": 415,
        "sand": 39,
        "brick": 10,
        "aggregate": 45,
        "steel": 61,
    }


def test_update_prices_sets_live_todaypricerates_source_label(monkeypatch):
    temp_dir = Path("tests/.tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = temp_dir / f"city_rates_{uuid4().hex}.csv"
    csv_path.write_text(
        "\n".join(
            [
                "city,state,tier,cost_mult,labour_mult,cement,sand,brick,aggregate,steel,verified,last_updated,source_label,source_url,notes",
                "coimbatore,Tamil Nadu,2,1.0,1.0,400,48,8,55,70,true,,manual label,https://property.todaypricerates.com/construction-building-materials-rate-in-Coimbatore,old notes",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "data.update_prices._fetch",
        lambda url: """
        OPC 53 Grade Cement: ₹390 – ₹440 per 50 kg bag
        River Sand: ₹4,500 – ₹6,200 per unit
        Red Clay Bricks: ₹9 – ₹12 per piece
        20mm Aggregate: ₹1,300 – ₹1,700 per ton
        Fe 500: ₹59,000 – ₹63,000 per metric ton
        """,
    )

    summary = update_prices(path=str(csv_path), dry_run=False, only_verified=True)

    assert len(summary["updated"]) == 1
    written = csv_path.read_text(encoding="utf-8")
    assert "todaypricerates.com live fetch" in written
    assert date.today().isoformat() in written
