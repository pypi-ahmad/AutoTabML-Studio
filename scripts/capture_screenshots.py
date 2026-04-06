"""Automated screenshot capture for AutoTabML Studio using Playwright.

Launches a real Streamlit session, loads a dataset, runs validation/profiling/
benchmark, then captures screenshots of every major page.

Usage:
    # 1. Start the Streamlit app in another terminal:
    #    streamlit run app/main.py --server.headless true --server.port 8501
    #
    # 2. Run this script:
    #    python scripts/capture_screenshots.py
    #
    # Or run with --launch to have the script start Streamlit automatically:
    #    python scripts/capture_screenshots.py --launch

Saves screenshots to docs/assets/screenshots/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

SCREENSHOTS_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets" / "screenshots"
APP_URL = "http://localhost:8501"
VIEWPORT = {"width": 1280, "height": 800}


def wait_for_streamlit(page: Page, timeout: int = 10_000) -> None:
    """Wait until the Streamlit app is loaded (main content visible)."""
    page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=timeout)
    # Let Streamlit finish rendering
    page.wait_for_timeout(1500)


def navigate_to_page(page: Page, label: str) -> None:
    """Click a sidebar radio option to navigate to a page."""
    sidebar = page.locator('[data-testid="stSidebar"]')
    # Streamlit radio options are rendered as label elements
    option = sidebar.locator(f'label:has-text("{label}")').first
    option.click()
    page.wait_for_timeout(2000)


def capture(page: Page, name: str, *, full_page: bool = False) -> Path:
    """Take a full-page screenshot and save it."""
    path = SCREENSHOTS_DIR / f"{name}.png"
    page.screenshot(path=str(path), full_page=full_page)
    print(f"  ✓ {name}.png ({path.stat().st_size // 1024} KB)")
    return path


def choose_streamlit_option(page: Page, label_text: str, option_text: str) -> None:
    """Select one option from a Streamlit selectbox/combobox by label text."""
    combobox = page.locator(f'input[role="combobox"][aria-label*="{label_text}"]').first
    combobox.click()
    page.wait_for_timeout(500)
    page.keyboard.press("Control+A")
    page.keyboard.type(option_text)
    page.wait_for_timeout(500)
    page.keyboard.press("Enter")
    page.wait_for_timeout(500)


def scroll_to_text(page: Page, text: str) -> None:
    """Scroll the page until the target text is visible near the viewport."""
    target = page.get_by_text(text).first
    target.scroll_into_view_if_needed(timeout=30000)
    page.wait_for_timeout(1000)


def upload_dataset(page: Page, csv_path: Path) -> None:
    """Load a dataset via the Local Path tab on Dataset Intake."""
    navigate_to_page(page, "Dataset Intake")
    page.wait_for_timeout(1000)

    # Click "Local Path" tab
    local_path_tab = page.locator('button[role="tab"]:has-text("Local Path")').first
    local_path_tab.click()
    page.wait_for_timeout(1000)

    # Fill the path text input and click Load
    path_input = page.get_by_label("Local dataset path").first
    path_input.fill(str(csv_path.resolve()))
    page.wait_for_timeout(500)

    load_btn = page.locator('button:has-text("Load Local Path")').first
    load_btn.click()
    page.get_by_text("Normalized Preview").wait_for(timeout=30000)
    page.wait_for_timeout(1500)


def run_validation(page: Page) -> None:
    """Navigate to Validation and run it."""
    navigate_to_page(page, "Validation")
    page.wait_for_timeout(1000)

    choose_streamlit_option(page, "Target column (optional)", "target")

    # Click "Run Validation" button
    run_btn = page.locator('button:has-text("Run Validation")').first
    if run_btn.is_visible():
        run_btn.click()
        page.get_by_text("Validation complete.").wait_for(timeout=30000)
        page.wait_for_timeout(1500)


def run_profiling(page: Page) -> None:
    """Navigate to Profiling and run it."""
    navigate_to_page(page, "Profiling")
    page.wait_for_timeout(1000)

    run_btn = page.locator('button:has-text("Generate Profile")').first
    if run_btn.is_visible():
        run_btn.click()
        page.get_by_text("Profiling complete.").wait_for(timeout=180000)
        page.wait_for_timeout(1500)


def run_benchmark(page: Page) -> None:
    """Navigate to Benchmark and run it."""
    navigate_to_page(page, "Benchmark")
    page.wait_for_timeout(1000)

    choose_streamlit_option(page, "Target column", "target")
    choose_streamlit_option(page, "Task type", "regression")

    advanced = page.get_by_text("Advanced options").first
    advanced.click()
    page.wait_for_timeout(500)
    page.get_by_label("Include models (comma-separated names)").first.fill(
        "DummyRegressor,LinearRegression"
    )
    page.wait_for_timeout(500)

    run_btn = page.locator('button:has-text("Run Benchmark")').first
    if run_btn.is_visible():
        run_btn.click()
        page.get_by_text("Benchmark complete.").wait_for(timeout=120000)
        page.wait_for_timeout(1500)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture AutoTabML Studio screenshots")
    parser.add_argument("--launch", action="store_true", help="Auto-launch Streamlit server")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit port")
    parser.add_argument("--dataset", type=str, default=None, help="Path to CSV dataset to load")
    parser.add_argument("--skip-workflows", action="store_true", help="Skip validation/profiling/benchmark runs")
    args = parser.parse_args()

    global APP_URL
    APP_URL = f"http://localhost:{args.port}"
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Find a dataset
    demo_csv = Path(args.dataset) if args.dataset else None
    if demo_csv is None:
        # Try to find a good small dataset
        candidates = [
            Path(__file__).resolve().parent.parent / "datasets" / "sklearn" / "Diabetes" / "diabetes.csv",
            Path(__file__).resolve().parent.parent / "datasets" / "sklearn" / "California_Housing",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.suffix == ".csv":
                demo_csv = candidate
                break
            elif candidate.is_dir():
                csvs = list(candidate.glob("*.csv"))
                if csvs:
                    demo_csv = csvs[0]
                    break

    server_proc = None
    if args.launch:
        print(f"Launching Streamlit on port {args.port}...")
        server_proc = subprocess.Popen(
            [
                sys.executable, "-m", "streamlit", "run", "app/main.py",
                "--server.headless", "true",
                "--server.port", str(args.port),
                "--browser.gatherUsageStats", "false",
            ],
            cwd=str(Path(__file__).resolve().parent.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("Waiting for Streamlit to start...")
        time.sleep(8)

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(
                viewport=VIEWPORT,
                device_scale_factor=2,  # Retina-quality screenshots
            )
            page = context.new_page()

            print(f"\nConnecting to {APP_URL}...")
            page.goto(APP_URL, wait_until="networkidle")
            wait_for_streamlit(page)
            print("Streamlit app loaded.\n")

            # ── Dataset Intake ─────────────────────────────────────────
            print("Capturing Dataset Intake...")
            if demo_csv and demo_csv.exists():
                upload_dataset(page, demo_csv)
                print(f"  Loaded: {demo_csv.name}")
            else:
                navigate_to_page(page, "Dataset Intake")
                page.wait_for_timeout(1500)
                print("  No dataset uploaded (use --dataset to specify)")
            capture(page, "dataset-intake")

            # ── Dashboard ──────────────────────────────────────────────
            print("Capturing Dashboard...")
            navigate_to_page(page, "Dashboard")
            page.wait_for_timeout(2000)
            capture(page, "dashboard-overview")

            # ── Validation ─────────────────────────────────────────────
            print("Capturing Validation...")
            if not args.skip_workflows:
                run_validation(page)
            else:
                navigate_to_page(page, "Validation")
                page.wait_for_timeout(1500)
            capture(page, "validation-summary")

            # ── Profiling ──────────────────────────────────────────────
            print("Capturing Profiling...")
            if not args.skip_workflows:
                run_profiling(page)
            else:
                navigate_to_page(page, "Profiling")
                page.wait_for_timeout(1500)
            scroll_to_text(page, "Missing %")
            capture(page, "profiling-report")

            # ── Benchmark ──────────────────────────────────────────────
            print("Capturing Benchmark...")
            if not args.skip_workflows:
                run_benchmark(page)
            else:
                navigate_to_page(page, "Benchmark")
                page.wait_for_timeout(1500)
            scroll_to_text(page, "Leaderboard")
            capture(page, "benchmark-leaderboard", full_page=True)

            # ── Experiment ─────────────────────────────────────────────
            print("Capturing Experiment...")
            navigate_to_page(page, "Experiment")
            page.wait_for_timeout(1500)
            capture(page, "experiment-lab")

            # ── Prediction ─────────────────────────────────────────────
            print("Capturing Prediction...")
            navigate_to_page(page, "Prediction")
            page.wait_for_timeout(1500)
            capture(page, "prediction-center")

            # ── History ────────────────────────────────────────────────
            print("Capturing History...")
            navigate_to_page(page, "History")
            page.wait_for_timeout(1500)
            capture(page, "history-view")

            # ── Compare ────────────────────────────────────────────────
            print("Capturing Compare...")
            navigate_to_page(page, "Compare")
            page.wait_for_timeout(1500)
            capture(page, "compare-view")

            # ── Registry ───────────────────────────────────────────────
            print("Capturing Registry...")
            navigate_to_page(page, "Registry")
            page.wait_for_timeout(1500)
            capture(page, "registry-view")

            # ── Settings ───────────────────────────────────────────────
            print("Capturing Settings...")
            navigate_to_page(page, "Settings")
            page.wait_for_timeout(1500)
            capture(page, "settings-view")

            browser.close()

        print(f"\n✅ All screenshots saved to {SCREENSHOTS_DIR.relative_to(Path.cwd())}/")
        print(f"   {len(list(SCREENSHOTS_DIR.glob('*.png')))} PNG files total")

    finally:
        if server_proc:
            server_proc.terminate()
            server_proc.wait(timeout=5)
            print("Streamlit server stopped.")


if __name__ == "__main__":
    main()
