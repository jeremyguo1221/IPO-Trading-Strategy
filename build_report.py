"""Inject analysis/report_data.json into report_template.html -> ipo_trends_report.html."""
from pathlib import Path

HERE = Path(__file__).parent
data = (HERE / "analysis" / "report_data.json").read_text(encoding="utf-8")
html = (HERE / "report_template.html").read_text(encoding="utf-8").replace("/*__DATA__*/null", data)
(HERE / "ipo_trends_report.html").write_text(html, encoding="utf-8")
print("wrote ipo_trends_report.html")
