# Asset Plan

This directory is reserved for real screenshots, demo media, and social preview assets.

Real screenshots and social preview graphics are now committed.

## Structure

- `screenshots/`: product screenshots used in the README or docs once they exist
- `social-preview/`: GitHub/social preview assets
- `demo/`: short clips, walkthrough notes, or demo-specific supporting media

## Current State

- `screenshots/` contains live captures from a real Streamlit session on April 6, 2026
- `social-preview/` contains the repo social preview image
- `demo/` remains the place for a future short walkthrough recording

## Recommended Screenshot Filenames

- `dashboard-overview.png`
- `validation-summary.png`
- `profiling-report.png`
- `benchmark-leaderboard.png`
- `experiment-lab.png`
- `prediction-center.png`
- `history-view.png`
- `registry-view.png`

## Capture Guidance

- use real data and real UI states
- avoid capturing secrets, local usernames, or machine-specific paths if possible
- prefer one screenshot per page/state rather than stitched fake composites
- keep the same browser window size across captures for consistency

## Automation

- `scripts/capture_screenshots.py`: repeatable Playwright-based capture flow for refreshing the screenshot set against a live local Streamlit instance

See the per-directory notes for more specific capture suggestions.
