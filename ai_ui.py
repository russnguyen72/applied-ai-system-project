"""Streamlit UI for the AI assistant."""
from __future__ import annotations

import streamlit as st

from ai_assistant import AssistantController, AssistantState
from llm_backend import OllamaBackend
from pawpal_system import Owner

_BORDER_COLORS = {
    AssistantState.AWAITING_CLARIFICATION: "#f59e0b",
    AssistantState.READY_TO_REVIEW: "#16a34a",
    AssistantState.APPLIED: "#16a34a",
    AssistantState.ERROR: "#dc2626",
}


def _inject_border_css(state: AssistantState) -> None:
    color = _BORDER_COLORS.get(state)
    if color is None:
        return
    st.markdown(
        f"""
        <style>
        .st-key-ai_assistant_box {{
            border: 2px solid {color} !important;
            border-radius: 10px !important;
            padding: 0.25rem !important;
            transition: border-color 0.25s ease-in-out;
        }}
        .st-key-ai_assistant_box [data-testid="stVerticalBlockBorderWrapper"] {{
            border-color: {color} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _inject_global_ui_css() -> None:
    """Swaps Streamlit's running-man top-right indicator for a clean CSS spinner."""
    st.markdown(
        """
        <style>
        @keyframes pawpal-spin {
            to { transform: rotate(360deg); }
        }
        [data-testid="stStatusWidget"] [data-testid="stIcon"],
        [data-testid="stStatusWidget"] svg,
        [data-testid="stStatusWidget"] img {
            display: none !important;
        }
        [data-testid="stStatusWidget"]::before {
            content: "";
            display: inline-block;
            width: 14px;
            height: 14px;
            margin-right: 8px;
            vertical-align: -2px;
            border: 2px solid #e5e7eb;
            border-top-color: #3b82f6;
            border-radius: 50%;
            animation: pawpal-spin 0.8s linear infinite;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _get_controller() -> AssistantController:
    if "ai_controller" not in st.session_state:
        st.session_state.ai_controller = AssistantController(backend=OllamaBackend())
    return st.session_state.ai_controller


def _render_health_banner(controller: AssistantController) -> None:
    if st.session_state.get("ai_health_ok"):
        return
    ok, message = controller.backend.health_check()
    st.session_state.ai_health_ok = ok
    st.session_state.ai_health_message = message
    if not ok:
        st.warning(f"AI assistant unavailable: {message}")


def _run_with_status(
    controller: AssistantController,
    show_thinking: bool,
    invoke,
) -> None:
    """Wraps a controller call in st.status, wiring progress + thinking callbacks."""
    with st.status("Starting…", expanded=show_thinking) as status:
        thinking_box = st.empty() if show_thinking else None
        thinking_buf = {"text": ""}

        def on_progress(msg: str) -> None:
            status.update(label=msg)

        def on_thinking(delta: str) -> None:
            if thinking_box is None:
                return
            thinking_buf["text"] += delta
            thinking_box.markdown(f"💭 _{thinking_buf['text']}_")

        invoke(on_progress, on_thinking, show_thinking)

        if controller.state == AssistantState.ERROR:
            status.update(state="error", label=controller.error_message or "Error")
        elif controller.state == AssistantState.AWAITING_CLARIFICATION:
            status.update(state="complete", label="Waiting for your reply")
        else:
            status.update(state="complete", label="Done")


def _render_idle(controller: AssistantController) -> None:
    st.caption(
        "Describe what you want in plain language. The assistant will stage pets and tasks for your review."
    )
    st.text_area(
        "Your request",
        key="ai_prompt_input",
        placeholder=(
            "Example: I bought 100 servings of dog food for my two dogs Max and Sherry. "
            "They eat twice a day at 8am and 6pm. Remind me to buy more food before we run out."
        ),
        height=120,
    )
    existing_names = [p.name for p in st.session_state.owner.pets]
    if st.button("Generate plan", type="primary", use_container_width=True):
        text = st.session_state.get("ai_prompt_input", "").strip()
        if text:
            show_thinking = True

            def invoke(on_progress, on_thinking, show):
                controller.submit_prompt(
                    text,
                    existing_pet_names=existing_names,
                    progress_callback=on_progress,
                    thinking_callback=on_thinking,
                    thinking_enabled=show,
                )

            _run_with_status(controller, show_thinking, invoke)
            st.rerun()


def _render_clarification(controller: AssistantController) -> None:
    # Clear any stale reply text from a previous clarification turn BEFORE the
    # widget is instantiated (Streamlit forbids mutating a widget's session_state
    # key after it renders).
    if st.session_state.pop("ai_clear_clarification", False):
        st.session_state.pop("ai_clarification_input", None)

    st.info(f"**The assistant needs more info:** {controller.pending_question}")
    st.text_input("Your reply", key="ai_clarification_input")

    cols = st.columns(2)
    with cols[0]:
        reply_clicked = st.button("Reply", type="primary", use_container_width=True)
    with cols[1]:
        cancel_clicked = st.button("Cancel", use_container_width=True)

    # Handle actions OUTSIDE the columns so the status panel spans full width.
    if reply_clicked:
        reply = st.session_state.get("ai_clarification_input", "").strip()
        if reply:
            show_thinking = True

            def invoke(on_progress, on_thinking, show):
                controller.submit_clarification(
                    reply,
                    progress_callback=on_progress,
                    thinking_callback=on_thinking,
                    thinking_enabled=show,
                )

            _run_with_status(controller, show_thinking, invoke)
            st.session_state.ai_clear_clarification = True
            st.rerun()
    elif cancel_clicked:
        controller.discard()
        st.rerun()


def _staged_pet_is_selected(controller: AssistantController, pet_name: str) -> bool:
    for staged in controller.staged_plan.pets:
        if staged.name == pet_name:
            return staged.selected
    return False


def _render_plan_review(controller: AssistantController, owner: Owner) -> None:
    plan = controller.staged_plan
    if plan.summary:
        st.markdown(f"**Summary:** {plan.summary}")

    if plan.pets:
        st.markdown("**Proposed pets**")
        for i, p in enumerate(plan.pets):
            label = f"{p.name} ({p.animal_type})"
            if p.last_vet_visit:
                label += f" — last vet: {p.last_vet_visit.isoformat()}"
            checked = st.checkbox(label, value=p.selected, key=f"ai_pet_{i}")
            plan.pets[i].selected = checked
    else:
        st.caption("No new pets staged.")

    if plan.tasks:
        st.markdown("**Proposed tasks**")
        for i, t in enumerate(plan.tasks):
            pet_exists_on_owner = any(p.name == t.pet_name for p in owner.pets)
            pet_selected_in_plan = _staged_pet_is_selected(controller, t.pet_name)
            can_apply = pet_exists_on_owner or pet_selected_in_plan

            label = (
                f"**{t.pet_name}** — {t.task_name} @ {t.scheduled_time.strftime('%H:%M')}, "
                f"every {t.frequency_days}d"
            )
            if t.start_in_days > 0:
                label += f" (starts in {t.start_in_days} days)"
            if t.description:
                label += f"  \n_{t.description}_"

            checked = st.checkbox(
                label,
                value=t.selected and can_apply,
                key=f"ai_task_{i}",
                disabled=not can_apply,
            )
            plan.tasks[i].selected = checked and can_apply
            if not can_apply:
                st.caption(
                    f"Disabled: pet '{t.pet_name}' isn't selected or doesn't exist yet."
                )
    else:
        st.caption("No new tasks staged.")

    cols = st.columns(2)
    with cols[0]:
        any_selected = (
            any(p.selected for p in plan.pets) or any(t.selected for t in plan.tasks)
        )
        if st.button(
            "Apply selected",
            type="primary",
            use_container_width=True,
            disabled=not any_selected,
        ):
            controller.apply_to_owner(owner)
            st.rerun()
    with cols[1]:
        if st.button("Discard", use_container_width=True):
            controller.discard()
            st.rerun()


def _render_applied(controller: AssistantController, owner: Owner) -> None:
    st.success("Plan applied. Review the Pets and Build Schedule sections below.")
    conflicts = owner.scheduler.get_scheduling_conflicts()
    if conflicts:
        st.markdown("**New scheduling conflicts detected:**")
        for c in conflicts:
            st.warning(c)
    if st.button("New request", use_container_width=True):
        controller.reset()
        st.rerun()


def _render_error(controller: AssistantController) -> None:
    st.error(controller.error_message or "Unknown error.")
    if st.button("Reset", use_container_width=True):
        controller.reset()
        st.rerun()


def render_assistant(owner: Owner) -> None:
    """Renders the full AI assistant UI block."""
    _inject_global_ui_css()
    controller = _get_controller()
    _inject_border_css(controller.state)

    with st.container(border=True, key="ai_assistant_box"):
        _render_health_banner(controller)

        state = controller.state
        if state in (AssistantState.IDLE, AssistantState.PLANNING):
            _render_idle(controller)
        elif state == AssistantState.AWAITING_CLARIFICATION:
            _render_clarification(controller)
        elif state == AssistantState.READY_TO_REVIEW:
            _render_plan_review(controller, owner)
        elif state == AssistantState.APPLIED:
            _render_applied(controller, owner)
        elif state == AssistantState.ERROR:
            _render_error(controller)
