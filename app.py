import json
import uuid
from datetime import time

import streamlit as st

from ai_ui import render_assistant
from pawpal_system import Owner, Pet, Task

st.set_page_config(page_title="PawPal+", page_icon="🐾", layout="centered")

st.title("🐾 PawPal+")

st.markdown(
    """
Welcome to the PawPal+ starter app.

This file is intentionally thin. It gives you a working Streamlit app so you can start quickly,
but **it does not implement the project logic**. Your job is to design the system and build it.

Use this app as your interactive demo once your backend classes/functions exist.
"""
)

with st.expander("Scenario", expanded=True):
    st.markdown(
        """
**PawPal+** is a pet care planning assistant. It helps a pet owner plan care tasks
for their pet(s) based on constraints like time, priority, and preferences.

You will design and implement the scheduling logic and connect it to this Streamlit UI.
"""
    )

with st.expander("What you need to build", expanded=True):
    st.markdown(
        """
At minimum, your system should:
- Represent pet care tasks (what needs to happen, how long it takes, priority)
- Represent the pet and the owner (basic info and preferences)
- Build a plan/schedule for a day that chooses and orders tasks based on constraints
- Explain the plan (why each task was chosen and when it happens)
"""
    )

st.divider()

# ── One-time initialization ────────────────────────────────────────────────
if "owner" not in st.session_state:
    from datetime import date
    from pawpal_system import TaskStatus

    st.session_state.owner = Owner(id=str(uuid.uuid4()))

    mochi = Pet(id=str(uuid.uuid4()), name="Mochi", animal_type="dog")
    mochi.add_task(Task(
        id=str(uuid.uuid4()), name="Morning Walk",
        description="30 min walk around the block",
        scheduled_time=time(7, 30), frequency_days=1,
    ))
    mochi.add_task(Task(
        id=str(uuid.uuid4()), name="Flea Treatment",
        description="Apply monthly flea and tick treatment",
        scheduled_time=time(9, 0), frequency_days=30,
        status=TaskStatus.COMPLETED,
        next_due_date=date.today(),
    ))
    mochi.add_task(Task(
        id=str(uuid.uuid4()), name="Evening Walk",
        description="15 min walk before sundown",
        scheduled_time=time(17, 0), frequency_days=1,
    ))

    whiskers = Pet(id=str(uuid.uuid4()), name="Whiskers", animal_type="cat")
    whiskers.add_task(Task(
        id=str(uuid.uuid4()), name="Breakfast",
        description="Half a can of wet food",
        scheduled_time=time(9, 0), frequency_days=1,  # conflicts with Mochi's Flea Treatment
    ))
    whiskers.add_task(Task(
        id=str(uuid.uuid4()), name="Lunchtime Play",
        description="15 min with the feather wand",
        scheduled_time=time(12, 0), frequency_days=1,
        status=TaskStatus.COMPLETED,
        next_due_date=date.today(),
    ))
    whiskers.add_task(Task(
        id=str(uuid.uuid4()), name="Dinner",
        description="Half a can of wet food",
        scheduled_time=time(18, 30), frequency_days=1,
    ))

    st.session_state.owner.add_pet(mochi)
    st.session_state.owner.add_pet(whiskers)

if "schedule" not in st.session_state:
    st.session_state.schedule = None


owner: Owner = st.session_state.owner

# ── AI Assistant ───────────────────────────────────────────────────────────
st.subheader("AI Assistant")
render_assistant(owner)
st.divider()

# ── Owner & Pets ───────────────────────────────────────────────────────────
st.subheader("Quick Demo Inputs")

st.text_input("Owner name", key="owner_name", value="Jordan")

# ── Save / Load ────────────────────────────────────────────────────────────
st.markdown("### Save / Load")

@st.dialog("Load save file")
def _load_dialog():
    st.caption("Upload a previously saved PawPal+ JSON file.")
    uploaded = st.file_uploader("", type=["json"], label_visibility="collapsed")
    confirm_col, cancel_col = st.columns(2)
    with confirm_col:
        if st.button("Load", disabled=uploaded is None, type="primary", use_container_width=True):
            try:
                loaded_owner = Owner.from_dict(json.loads(uploaded.read().decode("utf-8")))
                st.session_state.owner = loaded_owner
                st.session_state.schedule = None
                st.rerun()
            except ValueError as e:
                st.error(f"Could not load file: {e}")
    with cancel_col:
        if st.button("Cancel", use_container_width=True):
            st.rerun()

save_col, load_col = st.columns(2)
with save_col:
    st.download_button(
        label="Download save file",
        data=owner.to_json(),
        file_name="pawpal_save.json",
        mime="application/json",
        use_container_width=True,
    )
with load_col:
    if st.button("Load from file", use_container_width=True):
        _load_dialog()

st.markdown("### Pets")
st.caption("Pets currently being tracked.")

if owner.pets:
    st.table([
        {"Name": p.name, "Species": p.animal_type, "Tasks": len(p.tasks)}
        for p in owner.pets
    ])
else:
    st.info("No pets yet. Add one below.")

def _add_pet():
    name = st.session_state.new_pet_name.strip()
    if name:
        owner.add_pet(Pet(
            id=str(uuid.uuid4()),
            name=name,
            animal_type=st.session_state.new_pet_species,
        ))
        st.session_state.schedule = None

col_a, col_b = st.columns(2)
with col_a:
    st.text_input("Pet name", key="new_pet_name", value="")
with col_b:
    st.selectbox("Species", ["dog", "cat", "other"], key="new_pet_species")

st.button("Add pet", on_click=_add_pet)

# ── Tasks ──────────────────────────────────────────────────────────────────
st.markdown("### Tasks")
st.caption("Add tasks and assign them to one or all pets.")

if owner.pets:
    assign_options = [p.name for p in owner.pets] + ["All pets"]

    def _add_task():
        targets = (
            owner.pets if st.session_state.task_assign_to == "All pets"
            else [p for p in owner.pets if p.name == st.session_state.task_assign_to]
        )
        for target_pet in targets:
            target_pet.add_task(Task(
                id=str(uuid.uuid4()),
                name=st.session_state.task_name,
                description=st.session_state.task_desc,
                scheduled_time=st.session_state.task_time,
                frequency_days=int(st.session_state.task_freq),
            ))
        st.session_state.schedule = None

    st.selectbox("Assign to", assign_options, key="task_assign_to")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.text_input("Task name", key="task_name", value="Morning walk")
    with col2:
        st.text_input("Description", key="task_desc", value="Walk around the block")
    with col3:
        st.time_input("Scheduled time", key="task_time", value=time(8, 0))

    st.number_input(
        "Repeat every N days (0 = one-time)",
        key="task_freq", min_value=0, max_value=365, value=1,
    )

    if st.button("Add task"):
        _add_task()
else:
    st.info("Add a pet above before adding tasks.")

st.divider()

# ── Build Schedule ─────────────────────────────────────────────────────────
st.subheader("Build Schedule")
st.caption("Generates a time-ordered schedule across all pets.")

if owner.pets:
    def _generate_schedule():
        st.session_state.schedule = owner.scheduler.get_tasks_sorted_by_time()

    if st.button("Generate schedule"):
        _generate_schedule()

    if st.session_state.schedule is not None:
        if st.session_state.schedule:
            owner_label = st.session_state.get("owner_name", "")
            st.success(f"Schedule for **{owner_label}**'s pets")

            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                st.selectbox("Filter by pet", ["All pets"] + [p.name for p in owner.pets], key="task_filter_pet")
            with filter_col2:
                st.selectbox("Filter by status", ["All", "Pending", "Completed"], key="task_filter_status")

            filter_pet    = st.session_state.get("task_filter_pet", "All pets")
            filter_status = st.session_state.get("task_filter_status", "All")

            display_schedule = [
                (p, t) for p, t in st.session_state.schedule
                if (filter_pet    == "All pets"   or p.name == filter_pet)
                if (filter_status == "All"        or t.status.value.lower() == filter_status.lower())
            ]

            if display_schedule:
                st.table([
                    {
                        "Pet": p.name,
                        "Time": t.scheduled_time.strftime("%H:%M"),
                        "Task": t.name,
                        "Description": t.description,
                        "Status": t.status.value,
                        "Next due": str(t.next_due_date),
                    }
                    for p, t in display_schedule
                ])
            else:
                st.info("No tasks match the current filter.")

            conflicts = owner.scheduler.get_scheduling_conflicts()
            if conflicts:
                for conflict in conflicts:
                    st.warning(f"Time conflict: {conflict}")
        else:
            st.warning("No tasks to schedule for this selection.")
else:
    st.info("Add a pet above before generating a schedule.")
