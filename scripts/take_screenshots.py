"""Capture 1920x1080 screenshots of the running app (streamlit run app.py) with Playwright.

Usage:  python scripts/take_screenshots.py [out_dir]
"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "docs" / "media"
out.mkdir(parents=True, exist_ok=True)
URL = "http://localhost:8501"


def pick(pg, name):
    pg.locator("[data-testid='stSelectbox']").first.click()
    pg.get_by_role("option", name=name).click()
    pg.wait_for_timeout(3500)


with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1920, "height": 1080}, color_scheme="dark")
    pg.goto(URL)
    pg.wait_for_selector("text=Predicted defect", timeout=90000)
    pick(pg, "inclusion_3.jpg")
    pg.screenshot(path=str(out / "app_single_reject.png"))
    pick(pg, "patches_91.jpg")
    pg.screenshot(path=str(out / "app_single_accept.png"))
    # simulate defocus on a crazing sample
    pick(pg, "crazing_78.jpg")
    s = pg.get_by_role("slider").last
    s.focus()
    for _ in range(6):
        s.press("ArrowRight")
    pg.wait_for_timeout(4000)
    pg.screenshot(path=str(out / "app_single_defocus.png"))
    for _ in range(12):
        s.press("ArrowLeft")
    pg.wait_for_timeout(2500)
    pg.get_by_role("tab", name="Batch inspection log").click()
    pg.get_by_role("button", name="Run inspection").click()
    pg.wait_for_selector("text=Download inspection log", timeout=60000)
    pg.wait_for_timeout(2500)
    pg.screenshot(path=str(out / "app_batch.png"))
    b.close()
print("ok")
