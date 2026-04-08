"""Shared beginner-workflow banner for the main 5-step path."""

from __future__ import annotations

import streamlit as st

from app.pages.dataset_workspace import go_to_page

# ── Canonical beginner path ────────────────────────────────────────────
WORKFLOW_STEPS: list[dict[str, str]] = [
    {
        "number": "1",
        "label": "Load Data",
        "page": "Load Data",
        "icon": "📥",
        "short": "Upload a CSV / Excel file or load an example dataset.",
        "what_happens": "Your file is cleaned and loaded into memory so every page can use it.",
    },
    {
        "number": "2",
        "label": "Check Quality",
        "page": "Validation",
        "icon": "✅",
        "optional": True,
        "short": "Spot missing values, duplicates, and data-type issues before modeling.",
        "what_happens": "Automated checks run on your data and flag anything that might hurt model accuracy.",
    },
    {
        "number": "3",
        "label": "Find Best Model",
        "page": "Quick Benchmark",
        "icon": "🏁",
        "short": "Test dozens of algorithms in one click to find the best starting point.",
        "what_happens": "Each algorithm is trained and scored — you get a ranked leaderboard of results.",
    },
    {
        "number": "4",
        "label": "Train & Tune",
        "page": "Train & Tune",
        "icon": "🧪",
        "short": "Fine-tune the best algorithm and save a production-ready model.",
        "what_happens": "The system compares, tunes, and saves the winning model ready for predictions.",
        "alternatives": [
            {"label": "FLAML AutoML", "page": "FLAML AutoML", "icon": "🔥", "short": "Or use Microsoft FLAML for fast, automated model selection."},
        ],
    },
    {
        "number": "5",
        "label": "Predict",
        "page": "Predictions",
        "icon": "🔮",
        "short": "Use your trained model to score new data.",
        "what_happens": "Upload new rows and get predictions instantly — one at a time or in bulk.",
    },
]


def render_workflow_banner(current_step: int) -> None:
    """Render a horizontal step indicator showing position in the beginner path.

    Parameters
    ----------
    current_step:
        1-based index into *WORKFLOW_STEPS* (1 = Load Data … 5 = Predict).
    """
    cols = st.columns(len(WORKFLOW_STEPS))
    for idx, (col, step) in enumerate(zip(cols, WORKFLOW_STEPS), start=1):
        num = step["number"]
        label = step["label"]
        is_optional = step.get("optional", False)
        opt_tag = " *(optional)*" if is_optional else ""
        if idx == current_step:
            col.markdown(
                f"**`Step {num}`** · **{label}** ◀",
                help=f"You are here — step {num} of {len(WORKFLOW_STEPS)}.{' This step is optional.' if is_optional else ''}",
            )
        elif idx < current_step:
            if col.button(
                f"✓ {num}. {label}",
                key=f"wf_back_{step['page']}",
                use_container_width=True,
            ):
                go_to_page(step["page"])
        else:
            col.markdown(f"`{num}` · {label}{opt_tag}")


def render_next_step_hint(current_step: int) -> None:
    """Show a prominent *Next step* button after the current page's content.

    Does nothing if the user is already on the last step.
    """
    if current_step >= len(WORKFLOW_STEPS):
        return

    next_step = WORKFLOW_STEPS[current_step]  # 0-based: current_step points to next
    st.divider()
    c1, c2, _ = st.columns([2, 2, 4])
    c1.markdown(f"**Next step →** {next_step['icon']} {next_step['label']}")
    if c2.button(
        f"Go to {next_step['label']}",
        key=f"wf_next_{next_step['page']}",
        type="primary",
    ):
        go_to_page(next_step["page"])
