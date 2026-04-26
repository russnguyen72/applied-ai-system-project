"""Tests for the AI assistant controller. Uses a FakeBackend — no network calls."""
from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import date, time, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_assistant import (
    AssistantController,
    AssistantState,
    StagedPet,
    StagedPlan,
    StagedTask,
)
from pawpal_system import Owner, Pet


class FakeBackend:
    """Test double that returns scripted Ollama-shaped responses."""

    def __init__(
        self,
        scripted_responses: list[dict],
        thinking_texts: list[list[str]] | None = None,
    ):
        self.scripted_responses = list(scripted_responses)
        self.thinking_texts = list(thinking_texts) if thinking_texts is not None else []
        self.sent_messages: list[list[dict]] = []
        self.think_args: list[bool] = []
        self.stream_calls: int = 0

    def chat(self, messages: list[dict], tools: list[dict], think: bool = False) -> dict:
        self.sent_messages.append([dict(m) for m in messages])
        self.think_args.append(think)
        if not self.scripted_responses:
            return {"message": {"role": "assistant", "content": "", "tool_calls": []}}
        return self.scripted_responses.pop(0)

    def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict],
        think: bool = False,
        on_chunk=None,
    ) -> dict:
        self.stream_calls += 1
        self.sent_messages.append([dict(m) for m in messages])
        self.think_args.append(think)

        deltas = self.thinking_texts.pop(0) if self.thinking_texts else []
        if think and on_chunk is not None:
            for delta in deltas:
                on_chunk({"type": "thinking", "text": delta})

        if not self.scripted_responses:
            return {"message": {"role": "assistant", "content": "", "tool_calls": []}}
        return self.scripted_responses.pop(0)

    def health_check(self) -> tuple[bool, str]:
        return True, "FakeBackend ready"


def _tool_call(name: str, args: dict) -> dict:
    return {"function": {"name": name, "arguments": args}}


def _response_with_calls(tool_calls: list[dict]) -> dict:
    return {"message": {"role": "assistant", "content": "", "tool_calls": tool_calls}}


# --- apply_to_owner tests ---


def test_apply_only_selected_pets():
    controller = AssistantController(backend=FakeBackend([]))
    controller.staged_plan = StagedPlan(
        pets=[
            StagedPet(name="Max", animal_type="dog", selected=True),
            StagedPet(name="Sherry", animal_type="dog", selected=False),
        ],
    )
    controller.state = AssistantState.READY_TO_REVIEW

    owner = Owner(id=str(uuid.uuid4()))
    controller.apply_to_owner(owner)

    assert [p.name for p in owner.pets] == ["Max"]
    assert controller.state == AssistantState.APPLIED


def test_apply_links_tasks_by_name():
    controller = AssistantController(backend=FakeBackend([]))
    controller.staged_plan = StagedPlan(
        pets=[
            StagedPet(name="Max", animal_type="dog"),
            StagedPet(name="Sherry", animal_type="dog"),
        ],
        tasks=[
            StagedTask(
                pet_name="Max", task_name="Morning feed", description="",
                scheduled_time=time(8, 0), frequency_days=1,
            ),
            StagedTask(
                pet_name="Sherry", task_name="Evening feed", description="",
                scheduled_time=time(18, 0), frequency_days=1,
            ),
        ],
    )
    controller.state = AssistantState.READY_TO_REVIEW

    owner = Owner(id=str(uuid.uuid4()))
    controller.apply_to_owner(owner)

    max_pet = next(p for p in owner.pets if p.name == "Max")
    sherry_pet = next(p for p in owner.pets if p.name == "Sherry")
    assert [t.name for t in max_pet.tasks] == ["Morning feed"]
    assert [t.name for t in sherry_pet.tasks] == ["Evening feed"]


def test_apply_respects_start_in_days():
    controller = AssistantController(backend=FakeBackend([]))
    controller.staged_plan = StagedPlan(
        pets=[StagedPet(name="Max", animal_type="dog")],
        tasks=[StagedTask(
            pet_name="Max", task_name="Buy food", description="",
            scheduled_time=time(9, 0), frequency_days=0, start_in_days=22,
        )],
    )
    controller.state = AssistantState.READY_TO_REVIEW

    owner = Owner(id=str(uuid.uuid4()))
    controller.apply_to_owner(owner)

    task = owner.pets[0].tasks[0]
    assert task.next_due_date == date.today() + timedelta(days=22)


def test_apply_skips_task_when_pet_unselected_and_missing():
    controller = AssistantController(backend=FakeBackend([]))
    controller.staged_plan = StagedPlan(
        pets=[StagedPet(name="Max", animal_type="dog", selected=False)],
        tasks=[StagedTask(
            pet_name="Max", task_name="Feed", description="",
            scheduled_time=time(8, 0), frequency_days=1,
        )],
    )
    controller.state = AssistantState.READY_TO_REVIEW

    owner = Owner(id=str(uuid.uuid4()))
    controller.apply_to_owner(owner)

    assert owner.pets == []


def test_apply_attaches_tasks_to_existing_pets():
    controller = AssistantController(backend=FakeBackend([]))
    controller.staged_plan = StagedPlan(
        tasks=[StagedTask(
            pet_name="Rex", task_name="Walk", description="",
            scheduled_time=time(7, 30), frequency_days=1,
        )],
    )
    controller.state = AssistantState.READY_TO_REVIEW

    owner = Owner(id=str(uuid.uuid4()))
    rex = Pet(id=str(uuid.uuid4()), name="Rex", animal_type="dog")
    owner.add_pet(rex)

    controller.apply_to_owner(owner)

    assert [t.name for t in rex.tasks] == ["Walk"]


def test_apply_skips_unselected_tasks():
    controller = AssistantController(backend=FakeBackend([]))
    controller.staged_plan = StagedPlan(
        pets=[StagedPet(name="Max", animal_type="dog")],
        tasks=[
            StagedTask(
                pet_name="Max", task_name="Feed", description="",
                scheduled_time=time(8, 0), frequency_days=1, selected=True,
            ),
            StagedTask(
                pet_name="Max", task_name="Walk", description="",
                scheduled_time=time(17, 0), frequency_days=1, selected=False,
            ),
        ],
    )
    controller.state = AssistantState.READY_TO_REVIEW

    owner = Owner(id=str(uuid.uuid4()))
    controller.apply_to_owner(owner)

    assert [t.name for t in owner.pets[0].tasks] == ["Feed"]


def test_apply_noop_when_state_not_ready():
    controller = AssistantController(backend=FakeBackend([]))
    controller.staged_plan = StagedPlan(pets=[StagedPet(name="Max", animal_type="dog")])
    controller.state = AssistantState.IDLE

    owner = Owner(id=str(uuid.uuid4()))
    controller.apply_to_owner(owner)

    assert owner.pets == []


# --- Turn loop / state transition tests ---


def test_submit_prompt_transitions_to_awaiting_on_clarification():
    backend = FakeBackend([
        _response_with_calls([_tool_call("ask_clarification", {"question": "What time?"})]),
    ])
    controller = AssistantController(backend=backend)
    controller.submit_prompt("Add Max")

    assert controller.state == AssistantState.AWAITING_CLARIFICATION
    assert controller.pending_question == "What time?"


def test_submit_prompt_transitions_to_ready_on_finalize():
    backend = FakeBackend([
        _response_with_calls([_tool_call("create_pet", {"name": "Max", "animal_type": "dog"})]),
        _response_with_calls([_tool_call("finalize_plan", {"summary": "Staged Max."})]),
    ])
    controller = AssistantController(backend=backend)
    controller.submit_prompt("Add Max the dog")

    assert controller.state == AssistantState.READY_TO_REVIEW
    assert [p.name for p in controller.staged_plan.pets] == ["Max"]
    assert controller.staged_plan.summary == "Staged Max."


def test_submit_clarification_resumes_loop():
    backend = FakeBackend([
        _response_with_calls([_tool_call("ask_clarification", {"question": "Feeding time?"})]),
        _response_with_calls([
            _tool_call("create_pet", {"name": "Felix", "animal_type": "cat"}),
            _tool_call("create_task", {
                "pet_name": "Felix", "task_name": "Feed", "description": "wet food",
                "scheduled_time": "08:00", "frequency_days": 1,
            }),
        ]),
        _response_with_calls([_tool_call("finalize_plan", {"summary": "done"})]),
    ])
    controller = AssistantController(backend=backend)
    controller.submit_prompt("Add Felix the cat, remind me to feed him")
    assert controller.state == AssistantState.AWAITING_CLARIFICATION

    controller.submit_clarification("8am every day")

    assert controller.state == AssistantState.READY_TO_REVIEW
    assert len(controller.staged_plan.pets) == 1
    assert len(controller.staged_plan.tasks) == 1
    assert controller.staged_plan.tasks[0].scheduled_time == time(8, 0)


def test_turn_loop_bounded():
    # Script enough ask_clarification responses that we won't run out, and assert
    # the controller stops after MAX_TURNS rather than looping forever.
    from ai_assistant import MAX_TURNS

    # ask_clarification terminates the loop, so to actually exhaust MAX_TURNS
    # we need a non-terminating tool repeated — use create_pet with duplicate
    # names (rejected but loop continues).
    backend = FakeBackend([
        _response_with_calls([_tool_call("create_pet", {"name": "Dup", "animal_type": "dog"})])
    ] * (MAX_TURNS + 2))
    controller = AssistantController(backend=backend)
    controller.submit_prompt("Make stuff up")

    # After MAX_TURNS of non-terminating calls, should be in ERROR.
    assert controller.state == AssistantState.ERROR
    assert "without finalizing" in (controller.error_message or "")


def test_invalid_time_returns_error_to_model_and_loop_continues():
    backend = FakeBackend([
        _response_with_calls([_tool_call("create_pet", {"name": "Max", "animal_type": "dog"})]),
        _response_with_calls([_tool_call("create_task", {
            "pet_name": "Max", "task_name": "Feed", "description": "",
            "scheduled_time": "banana", "frequency_days": 1,
        })]),
        _response_with_calls([_tool_call("create_task", {
            "pet_name": "Max", "task_name": "Feed", "description": "",
            "scheduled_time": "08:00", "frequency_days": 1,
        })]),
        _response_with_calls([_tool_call("finalize_plan", {"summary": "ok"})]),
    ])
    controller = AssistantController(backend=backend)
    controller.submit_prompt("Add Max, feed at ???")

    assert controller.state == AssistantState.READY_TO_REVIEW
    assert len(controller.staged_plan.tasks) == 1
    assert controller.staged_plan.tasks[0].scheduled_time == time(8, 0)


def test_create_task_rejects_unknown_pet():
    backend = FakeBackend([
        _response_with_calls([_tool_call("create_task", {
            "pet_name": "Ghost", "task_name": "Feed", "description": "",
            "scheduled_time": "08:00", "frequency_days": 1,
        })]),
        _response_with_calls([_tool_call("finalize_plan", {"summary": "empty"})]),
    ])
    controller = AssistantController(backend=backend)
    controller.submit_prompt("feed Ghost")

    assert controller.staged_plan.tasks == []


def test_create_pet_rejects_duplicate_of_existing():
    backend = FakeBackend([
        _response_with_calls([_tool_call("create_pet", {"name": "Rex", "animal_type": "dog"})]),
        _response_with_calls([_tool_call("finalize_plan", {"summary": "ok"})]),
    ])
    controller = AssistantController(backend=backend)
    controller.submit_prompt("Add Rex", existing_pet_names=["Rex"])

    assert controller.staged_plan.pets == []


def test_controller_recovers_from_backend_error():
    class BrokenBackend:
        def chat(self, messages, tools, think=False):
            from llm_backend import LLMBackendError
            raise LLMBackendError("daemon unreachable")

        def chat_stream(self, messages, tools, think=False, on_chunk=None):
            from llm_backend import LLMBackendError
            raise LLMBackendError("daemon unreachable")

        def health_check(self):
            return False, "daemon unreachable"

    controller = AssistantController(backend=BrokenBackend())
    controller.submit_prompt("hello")

    assert controller.state == AssistantState.ERROR
    assert "daemon unreachable" in (controller.error_message or "")


# --- End-to-end dog food scenario ---


def test_dog_food_scenario_end_to_end():
    """The core user scenario: 100 servings of dog food, two dogs eating twice a day."""
    backend = FakeBackend([
        _response_with_calls([
            _tool_call("create_pet", {"name": "Max", "animal_type": "dog"}),
            _tool_call("create_pet", {"name": "Sherry", "animal_type": "dog"}),
        ]),
        _response_with_calls([
            _tool_call("create_task", {
                "pet_name": "Max", "task_name": "Morning feed",
                "description": "Dog food breakfast",
                "scheduled_time": "08:00", "frequency_days": 1,
            }),
            _tool_call("create_task", {
                "pet_name": "Max", "task_name": "Evening feed",
                "description": "Dog food dinner",
                "scheduled_time": "18:00", "frequency_days": 1,
            }),
            _tool_call("create_task", {
                "pet_name": "Sherry", "task_name": "Morning feed",
                "description": "Dog food breakfast",
                "scheduled_time": "08:00", "frequency_days": 1,
            }),
            _tool_call("create_task", {
                "pet_name": "Sherry", "task_name": "Evening feed",
                "description": "Dog food dinner",
                "scheduled_time": "18:00", "frequency_days": 1,
            }),
            _tool_call("create_task", {
                "pet_name": "Max", "task_name": "Buy dog food",
                "description": "About 10 servings left; restock before running out.",
                "scheduled_time": "09:00", "frequency_days": 0, "start_in_days": 22,
            }),
        ]),
        _response_with_calls([
            _tool_call("finalize_plan", {
                "summary": "Staged Max and Sherry with twice-daily feeds and a buy-more-food reminder.",
            }),
        ]),
    ])

    controller = AssistantController(backend=backend)
    controller.submit_prompt(
        "I bought 100 servings of dog food for my two dogs Max and Sherry. "
        "They eat twice a day. Remind me to buy more before we run out."
    )

    assert controller.state == AssistantState.READY_TO_REVIEW
    assert [p.name for p in controller.staged_plan.pets] == ["Max", "Sherry"]
    assert len(controller.staged_plan.tasks) == 5

    reminder = next(t for t in controller.staged_plan.tasks if t.task_name == "Buy dog food")
    assert reminder.start_in_days == 22
    assert reminder.frequency_days == 0

    owner = Owner(id=str(uuid.uuid4()))
    controller.apply_to_owner(owner)

    assert controller.state == AssistantState.APPLIED
    assert len(owner.pets) == 2
    all_tasks = [t for p in owner.pets for t in p.tasks]
    assert len(all_tasks) == 5
    reminder_task = next(
        t for p in owner.pets for t in p.tasks if t.name == "Buy dog food"
    )
    assert reminder_task.next_due_date == date.today() + timedelta(days=22)


# --- Staged plan dataclass defaults ---


def test_staged_plan_defaults():
    plan = StagedPlan()
    assert plan.pets == []
    assert plan.tasks == []
    assert plan.summary == ""


def test_reset_clears_all_state():
    controller = AssistantController(backend=FakeBackend([]))
    controller.state = AssistantState.READY_TO_REVIEW
    controller.staged_plan = StagedPlan(pets=[StagedPet(name="X", animal_type="dog")])
    controller.pending_question = "?"
    controller.error_message = "bad"
    controller.messages = [{"role": "user", "content": "hi"}]
    controller.thinking_log = ["some thinking"]

    controller.reset()

    assert controller.state == AssistantState.IDLE
    assert controller.staged_plan.pets == []
    assert controller.pending_question is None
    assert controller.error_message is None
    assert controller.messages == []
    assert controller.thinking_log == []


# --- Progress & thinking callback tests ---


def test_progress_callback_receives_phases():
    backend = FakeBackend([
        _response_with_calls([_tool_call("create_pet", {"name": "Max", "animal_type": "dog"})]),
        _response_with_calls([_tool_call("finalize_plan", {"summary": "ok"})]),
    ])
    controller = AssistantController(backend=backend)

    events: list[str] = []
    controller.submit_prompt("Add Max the dog", progress_callback=events.append)

    assert events[0] == "Reading your request"
    assert "Thinking" in events
    assert "Adding pet: Max" in events
    assert "Finalizing plan" in events
    assert events[-1] == "Plan ready for review"


def test_progress_callback_labels_tool_calls():
    backend = FakeBackend([
        _response_with_calls([
            _tool_call("create_pet", {"name": "Max", "animal_type": "dog"}),
            _tool_call("create_task", {
                "pet_name": "Max", "task_name": "Morning feed", "description": "",
                "scheduled_time": "08:00", "frequency_days": 1,
            }),
        ]),
        _response_with_calls([_tool_call("finalize_plan", {"summary": "ok"})]),
    ])
    controller = AssistantController(backend=backend)

    events: list[str] = []
    controller.submit_prompt("Add Max and feed him", progress_callback=events.append)

    assert "Adding pet: Max" in events
    assert "Scheduling Morning feed for Max" in events


def test_progress_callback_on_clarification():
    backend = FakeBackend([
        _response_with_calls([_tool_call("ask_clarification", {"question": "time?"})]),
    ])
    controller = AssistantController(backend=backend)

    events: list[str] = []
    controller.submit_prompt("Add a cat Felix", progress_callback=events.append)

    assert "Preparing a question" in events
    assert events[-1] == "Waiting for your reply"


def test_progress_callback_on_backend_error():
    class BrokenBackend:
        def chat(self, messages, tools, think=False):
            from llm_backend import LLMBackendError
            raise LLMBackendError("down")

        def chat_stream(self, messages, tools, think=False, on_chunk=None):
            from llm_backend import LLMBackendError
            raise LLMBackendError("down")

        def health_check(self):
            return False, "down"

    controller = AssistantController(backend=BrokenBackend())
    events: list[str] = []
    controller.submit_prompt("hi", progress_callback=events.append)

    assert controller.state == AssistantState.ERROR
    assert any(e.startswith("Error:") for e in events)


def test_thinking_callback_receives_deltas_when_enabled():
    backend = FakeBackend(
        scripted_responses=[
            _response_with_calls([_tool_call("finalize_plan", {"summary": "ok"})]),
        ],
        thinking_texts=[["I'll ", "start by ", "adding Max."]],
    )
    controller = AssistantController(backend=backend)

    deltas: list[str] = []
    controller.submit_prompt(
        "plan it",
        thinking_callback=deltas.append,
        thinking_enabled=True,
    )

    assert deltas == ["I'll ", "start by ", "adding Max."]
    assert backend.think_args == [True]
    assert controller.thinking_log == ["I'll ", "start by ", "adding Max."]


def test_thinking_disabled_passes_think_false_and_fires_no_callback():
    backend = FakeBackend(
        scripted_responses=[
            _response_with_calls([_tool_call("finalize_plan", {"summary": "ok"})]),
        ],
        thinking_texts=[["should not be sent"]],
    )
    controller = AssistantController(backend=backend)

    deltas: list[str] = []
    controller.submit_prompt(
        "plan it",
        thinking_callback=deltas.append,
        thinking_enabled=False,
    )

    assert deltas == []
    assert backend.think_args == [False]
    assert controller.thinking_log == []


def test_callbacks_optional_is_noop():
    backend = FakeBackend([
        _response_with_calls([_tool_call("create_pet", {"name": "Max", "animal_type": "dog"})]),
        _response_with_calls([_tool_call("finalize_plan", {"summary": "ok"})]),
    ])
    controller = AssistantController(backend=backend)

    # No callbacks, no thinking — must not raise.
    controller.submit_prompt("Add Max the dog")

    assert controller.state == AssistantState.READY_TO_REVIEW
    assert [p.name for p in controller.staged_plan.pets] == ["Max"]


def test_submit_clarification_emits_progress():
    backend = FakeBackend([
        _response_with_calls([_tool_call("ask_clarification", {"question": "time?"})]),
        _response_with_calls([_tool_call("finalize_plan", {"summary": "ok"})]),
    ])
    controller = AssistantController(backend=backend)
    controller.submit_prompt("Add Felix")
    assert controller.state == AssistantState.AWAITING_CLARIFICATION

    events: list[str] = []
    controller.submit_clarification("8am daily", progress_callback=events.append)

    assert events[0] == "Processing your reply"
    assert "Finalizing plan" in events
    assert events[-1] == "Plan ready for review"


# --- apply_to_owner edge cases ---


def test_apply_does_not_duplicate_pet_already_in_owner():
    controller = AssistantController(backend=FakeBackend([]))
    controller.staged_plan = StagedPlan(
        pets=[StagedPet(name="Rex", animal_type="dog", selected=True)],
    )
    controller.state = AssistantState.READY_TO_REVIEW

    owner = Owner(id=str(uuid.uuid4()))
    rex = Pet(id=str(uuid.uuid4()), name="Rex", animal_type="dog")
    owner.add_pet(rex)

    controller.apply_to_owner(owner)

    assert len(owner.pets) == 1
    assert owner.pets[0] is rex


def test_apply_start_in_days_zero_is_today():
    controller = AssistantController(backend=FakeBackend([]))
    controller.staged_plan = StagedPlan(
        pets=[StagedPet(name="Max", animal_type="dog")],
        tasks=[StagedTask(
            pet_name="Max", task_name="Feed", description="",
            scheduled_time=time(8, 0), frequency_days=1, start_in_days=0,
        )],
    )
    controller.state = AssistantState.READY_TO_REVIEW

    owner = Owner(id=str(uuid.uuid4()))
    controller.apply_to_owner(owner)

    assert owner.pets[0].tasks[0].next_due_date == date.today()


def test_apply_negative_start_in_days_clamped_to_zero():
    controller = AssistantController(backend=FakeBackend([]))
    controller.staged_plan = StagedPlan(
        pets=[StagedPet(name="Max", animal_type="dog")],
        tasks=[StagedTask(
            pet_name="Max", task_name="Feed", description="",
            scheduled_time=time(8, 0), frequency_days=1, start_in_days=-3,
        )],
    )
    controller.state = AssistantState.READY_TO_REVIEW

    owner = Owner(id=str(uuid.uuid4()))
    controller.apply_to_owner(owner)

    assert owner.pets[0].tasks[0].next_due_date == date.today()


def test_apply_propagates_last_vet_visit():
    controller = AssistantController(backend=FakeBackend([]))
    vet_date = date(2024, 6, 15)
    controller.staged_plan = StagedPlan(
        pets=[StagedPet(name="Max", animal_type="dog", last_vet_visit=vet_date)],
    )
    controller.state = AssistantState.READY_TO_REVIEW

    owner = Owner(id=str(uuid.uuid4()))
    controller.apply_to_owner(owner)

    assert owner.pets[0].last_vet_visit == vet_date


def test_apply_empty_staged_plan_transitions_to_applied():
    controller = AssistantController(backend=FakeBackend([]))
    controller.staged_plan = StagedPlan()
    controller.state = AssistantState.READY_TO_REVIEW

    owner = Owner(id=str(uuid.uuid4()))
    controller.apply_to_owner(owner)

    assert controller.state == AssistantState.APPLIED
    assert owner.pets == []


# --- discard tests ---


def test_discard_resets_to_idle():
    controller = AssistantController(backend=FakeBackend([]))
    controller.state = AssistantState.READY_TO_REVIEW
    controller.staged_plan = StagedPlan(pets=[StagedPet(name="X", animal_type="dog")])
    controller.pending_question = "?"
    controller.error_message = "err"
    controller.messages = [{"role": "user", "content": "hi"}]

    controller.discard()

    assert controller.state == AssistantState.IDLE
    assert controller.staged_plan.pets == []
    assert controller.pending_question is None
    assert controller.error_message is None
    assert controller.messages == []


# --- submit_clarification edge cases ---


def test_submit_clarification_noop_when_wrong_state():
    backend = FakeBackend([
        _response_with_calls([_tool_call("finalize_plan", {"summary": "ok"})]),
    ])
    controller = AssistantController(backend=backend)
    controller.state = AssistantState.IDLE

    controller.submit_clarification("doesn't matter")

    assert controller.state == AssistantState.IDLE
    assert backend.stream_calls == 0


# --- Implicit finalize / no-tool-calls paths ---


def test_no_tool_calls_with_staged_pets_is_implicit_finalize():
    backend = FakeBackend([
        _response_with_calls([_tool_call("create_pet", {"name": "Max", "animal_type": "dog"})]),
        {"message": {"role": "assistant", "content": "Max is all set!", "tool_calls": []}},
    ])
    controller = AssistantController(backend=backend)
    controller.submit_prompt("Add Max")

    assert controller.state == AssistantState.READY_TO_REVIEW
    assert controller.staged_plan.summary == "Max is all set!"


def test_no_tool_calls_with_empty_plan_is_error():
    backend = FakeBackend([
        {"message": {"role": "assistant", "content": "I don't understand.", "tool_calls": []}},
    ])
    controller = AssistantController(backend=backend)
    controller.submit_prompt("???")

    assert controller.state == AssistantState.ERROR
    assert "without calling any tool" in (controller.error_message or "")


# --- Tool validation edge cases ---


def test_create_task_negative_frequency_days_is_rejected():
    backend = FakeBackend([
        _response_with_calls([_tool_call("create_pet", {"name": "Max", "animal_type": "dog"})]),
        _response_with_calls([_tool_call("create_task", {
            "pet_name": "Max", "task_name": "Feed", "description": "",
            "scheduled_time": "08:00", "frequency_days": -1,
        })]),
        _response_with_calls([_tool_call("finalize_plan", {"summary": "ok"})]),
    ])
    controller = AssistantController(backend=backend)
    controller.submit_prompt("Feed Max daily")

    assert controller.state == AssistantState.READY_TO_REVIEW
    assert controller.staged_plan.tasks == []


def test_create_task_negative_start_in_days_is_rejected():
    backend = FakeBackend([
        _response_with_calls([_tool_call("create_pet", {"name": "Max", "animal_type": "dog"})]),
        _response_with_calls([_tool_call("create_task", {
            "pet_name": "Max", "task_name": "Buy food", "description": "",
            "scheduled_time": "09:00", "frequency_days": 0, "start_in_days": -5,
        })]),
        _response_with_calls([_tool_call("finalize_plan", {"summary": "ok"})]),
    ])
    controller = AssistantController(backend=backend)
    controller.submit_prompt("Remind me to buy food eventually")

    assert controller.state == AssistantState.READY_TO_REVIEW
    assert controller.staged_plan.tasks == []


def test_create_task_for_existing_pet_name():
    backend = FakeBackend([
        _response_with_calls([_tool_call("create_task", {
            "pet_name": "Rex", "task_name": "Walk", "description": "",
            "scheduled_time": "07:30", "frequency_days": 1,
        })]),
        _response_with_calls([_tool_call("finalize_plan", {"summary": "ok"})]),
    ])
    controller = AssistantController(backend=backend)
    controller.submit_prompt("Walk Rex daily", existing_pet_names=["Rex"])

    assert controller.state == AssistantState.READY_TO_REVIEW
    assert len(controller.staged_plan.tasks) == 1
    assert controller.staged_plan.tasks[0].task_name == "Walk"


def test_create_pet_rejects_invalid_last_vet_date():
    backend = FakeBackend([
        _response_with_calls([_tool_call("create_pet", {
            "name": "Max", "animal_type": "dog", "last_vet_visit": "not-a-date",
        })]),
        _response_with_calls([_tool_call("create_pet", {
            "name": "Max", "animal_type": "dog",
        })]),
        _response_with_calls([_tool_call("finalize_plan", {"summary": "ok"})]),
    ])
    controller = AssistantController(backend=backend)
    controller.submit_prompt("Add Max (bad date)")

    assert controller.state == AssistantState.READY_TO_REVIEW
    assert len(controller.staged_plan.pets) == 1
    assert controller.staged_plan.pets[0].last_vet_visit is None


def test_unknown_tool_call_loops_with_error():
    backend = FakeBackend([
        _response_with_calls([_tool_call("nonexistent_tool", {"x": 1})]),
        _response_with_calls([_tool_call("create_pet", {"name": "Max", "animal_type": "dog"})]),
        _response_with_calls([_tool_call("finalize_plan", {"summary": "ok"})]),
    ])
    controller = AssistantController(backend=backend)
    controller.submit_prompt("Add Max")

    assert controller.state == AssistantState.READY_TO_REVIEW
    assert len(controller.staged_plan.pets) == 1


# --- Module-level helper tests ---


def test_parse_time_returns_time():
    from ai_assistant import _parse_time
    assert _parse_time("08:30") == time(8, 30)
    assert _parse_time("18:00") == time(18, 0)


def test_parse_date_returns_date():
    from ai_assistant import _parse_date
    assert _parse_date("2024-06-15") == date(2024, 6, 15)


# --- Controller __init__ defaults ---


def test_controller_init_defaults():
    controller = AssistantController(backend=FakeBackend([]))
    assert controller.state == AssistantState.IDLE
    assert controller.messages == []
    assert controller.staged_plan.pets == []
    assert controller.staged_plan.tasks == []
    assert controller.pending_question is None
    assert controller.error_message is None
    assert controller.existing_pet_names == []
    assert controller.thinking_enabled is False
    assert controller.thinking_log == []


# --- _emit (static) tests ---


def test_emit_with_none_callback_is_noop():
    AssistantController._emit("hello", None)


def test_emit_swallows_callback_exceptions():
    def raising_cb(_label):
        raise RuntimeError("boom")
    AssistantController._emit("hello", raising_cb)


def test_emit_invokes_callback():
    received: list[str] = []
    AssistantController._emit("hello", received.append)
    assert received == ["hello"]


# --- _label_for_tool_call (static) tests ---


def test_label_for_tool_call_all_branches():
    L = AssistantController._label_for_tool_call
    assert L("create_pet", {"name": "Max"}) == "Adding pet: Max"
    assert L("create_pet", {}) == "Adding pet: pet"
    assert L("create_task", {"pet_name": "Max", "task_name": "Walk"}) == "Scheduling Walk for Max"
    assert L("create_task", {}) == "Scheduling task for pet"
    assert L("ask_clarification", {}) == "Preparing a question"
    assert L("finalize_plan", {}) == "Finalizing plan"
    assert L("mystery", {}) == "Processing mystery"


# --- _handle_tool_call dispatch tests ---


def test_handle_tool_call_dispatches_create_pet():
    controller = AssistantController(backend=FakeBackend([]))
    result = controller._handle_tool_call("create_pet", {"name": "Max", "animal_type": "dog"})
    parsed = json.loads(result)
    assert parsed["ok"] is True
    assert [p.name for p in controller.staged_plan.pets] == ["Max"]


def test_handle_tool_call_unknown_tool_returns_error():
    controller = AssistantController(backend=FakeBackend([]))
    result = controller._handle_tool_call("nonexistent", {})
    parsed = json.loads(result)
    assert parsed["ok"] is False
    assert "unknown tool" in parsed["error"]


def test_handle_tool_call_swallows_exception_from_tool():
    controller = AssistantController(backend=FakeBackend([]))

    def boom(_args):
        raise RuntimeError("kaboom")
    controller._tool_create_pet = boom  # shadow instance attribute

    result = controller._handle_tool_call("create_pet", {})
    parsed = json.loads(result)
    assert parsed["ok"] is False
    assert "kaboom" in parsed["error"]


# --- _tool_ask_clarification tests ---


def test_tool_ask_clarification_sets_pending_question():
    controller = AssistantController(backend=FakeBackend([]))
    result = controller._tool_ask_clarification({"question": "When?"})
    parsed = json.loads(result)
    assert parsed["ok"] is True
    assert parsed["awaiting_user"] is True
    assert controller.pending_question == "When?"


def test_tool_ask_clarification_rejects_invalid():
    controller = AssistantController(backend=FakeBackend([]))
    for bad in ({}, {"question": ""}, {"question": "   "}, {"question": 42}):
        result = controller._tool_ask_clarification(bad)
        assert json.loads(result)["ok"] is False
    assert controller.pending_question is None


# --- _tool_finalize_plan tests ---


def test_tool_finalize_plan_sets_summary():
    controller = AssistantController(backend=FakeBackend([]))
    result = controller._tool_finalize_plan({"summary": "All done."})
    parsed = json.loads(result)
    assert parsed["ok"] is True
    assert controller.staged_plan.summary == "All done."


def test_tool_finalize_plan_defaults_when_missing_or_empty():
    controller = AssistantController(backend=FakeBackend([]))
    controller._tool_finalize_plan({})
    assert controller.staged_plan.summary == "Plan ready for review."

    controller2 = AssistantController(backend=FakeBackend([]))
    controller2._tool_finalize_plan({"summary": ""})
    assert controller2.staged_plan.summary == "Plan ready for review."

    controller3 = AssistantController(backend=FakeBackend([]))
    controller3._tool_finalize_plan({"summary": 123})
    assert controller3.staged_plan.summary == "Plan ready for review."


# --- _fold_clarification_answer tests ---


def test_fold_clarification_answer_rewrites_tool_message():
    controller = AssistantController(backend=FakeBackend([]))
    controller.pending_question = "What time?"
    controller.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Add Max"},
        {"role": "assistant", "content": "", "tool_calls": []},
        {
            "role": "tool",
            "name": "ask_clarification",
            "content": json.dumps({"ok": True, "awaiting_user": True}),
        },
    ]

    controller._fold_clarification_answer("8am")

    folded = controller.messages[-1]
    assert folded["role"] == "tool"
    assert folded["name"] == "ask_clarification"
    payload = json.loads(folded["content"])
    assert payload == {"ok": True, "question": "What time?", "answer": "8am"}
    # No extra user message appended.
    assert len(controller.messages) == 4


def test_fold_clarification_answer_finds_most_recent():
    controller = AssistantController(backend=FakeBackend([]))
    controller.pending_question = "Latest?"
    controller.messages = [
        {"role": "tool", "name": "ask_clarification", "content": "old"},
        {"role": "user", "content": "first reply"},
        {"role": "tool", "name": "create_pet", "content": "{}"},
        {"role": "tool", "name": "ask_clarification", "content": "newer"},
    ]

    controller._fold_clarification_answer("answer")

    # Only the newest match should be rewritten.
    assert controller.messages[0]["content"] == "old"
    assert controller.messages[2]["content"] == "{}"
    payload = json.loads(controller.messages[3]["content"])
    assert payload["answer"] == "answer"


def test_fold_clarification_answer_falls_back_when_no_tool_message():
    controller = AssistantController(backend=FakeBackend([]))
    controller.pending_question = "Q?"
    controller.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]

    controller._fold_clarification_answer("the reply")

    assert controller.messages[-1] == {"role": "user", "content": "the reply"}


def test_fold_clarification_answer_handles_missing_pending_question():
    controller = AssistantController(backend=FakeBackend([]))
    controller.pending_question = None
    controller.messages = [
        {"role": "tool", "name": "ask_clarification", "content": "old"},
    ]

    controller._fold_clarification_answer("answer")

    payload = json.loads(controller.messages[0]["content"])
    assert payload["question"] == ""
    assert payload["answer"] == "answer"


# --- Response parsing helper tests (_extract_*, _normalize_assistant_msg) ---


def test_extract_message_from_dict():
    response = {"message": {"role": "assistant", "content": "hi"}}
    assert AssistantController._extract_message(response) == {"role": "assistant", "content": "hi"}


def test_extract_message_handles_missing_and_non_dict():
    assert AssistantController._extract_message({}) == {}
    assert AssistantController._extract_message("not a dict") == {}
    assert AssistantController._extract_message({"message": None}) == {}


def test_extract_message_handles_model_dump_object():
    class FakeMsg:
        def model_dump(self):
            return {"role": "assistant", "content": "from dump"}
    assert AssistantController._extract_message({"message": FakeMsg()}) == {
        "role": "assistant", "content": "from dump",
    }


def test_extract_tool_calls():
    msg = {"tool_calls": [{"function": {"name": "x"}}]}
    assert AssistantController._extract_tool_calls(msg) == [{"function": {"name": "x"}}]
    assert AssistantController._extract_tool_calls({}) == []
    assert AssistantController._extract_tool_calls({"tool_calls": None}) == []


def test_extract_text():
    assert AssistantController._extract_text({"content": "hello"}) == "hello"
    assert AssistantController._extract_text({"content": None}) == ""
    assert AssistantController._extract_text({"content": 42}) == ""
    assert AssistantController._extract_text({}) == ""


def test_extract_name_and_args_dict_form():
    call = {"function": {"name": "create_pet", "arguments": {"name": "Max"}}}
    name, args = AssistantController._extract_name_and_args(call)
    assert name == "create_pet"
    assert args == {"name": "Max"}


def test_extract_name_and_args_string_arguments():
    call = {"function": {"name": "create_pet", "arguments": '{"name": "Max"}'}}
    name, args = AssistantController._extract_name_and_args(call)
    assert (name, args) == ("create_pet", {"name": "Max"})


def test_extract_name_and_args_invalid_json_string():
    call = {"function": {"name": "x", "arguments": "not-json"}}
    name, args = AssistantController._extract_name_and_args(call)
    assert name == "x"
    assert args == {}


def test_extract_name_and_args_top_level_fallback():
    call = {"name": "create_pet", "arguments": {"name": "Max"}}
    name, args = AssistantController._extract_name_and_args(call)
    assert name == "create_pet"
    assert args == {"name": "Max"}


def test_extract_name_and_args_model_dump_call():
    class FakeCall:
        def model_dump(self):
            return {"function": {"name": "create_pet", "arguments": {"name": "Max"}}}
    name, args = AssistantController._extract_name_and_args(FakeCall())
    assert name == "create_pet"
    assert args == {"name": "Max"}


def test_extract_name_and_args_model_dump_function():
    class FakeFn:
        def model_dump(self):
            return {"name": "create_pet", "arguments": {"name": "Max"}}
    name, args = AssistantController._extract_name_and_args({"function": FakeFn()})
    assert name == "create_pet"
    assert args == {"name": "Max"}


def test_extract_name_and_args_non_dict_args_returns_empty():
    call = {"function": {"name": "x", "arguments": ["not", "a", "dict"]}}
    _, args = AssistantController._extract_name_and_args(call)
    assert args == {}


def test_normalize_assistant_msg_basic():
    out = AssistantController._normalize_assistant_msg({"role": "assistant", "content": "hello"})
    assert out == {"role": "assistant", "content": "hello"}


def test_normalize_assistant_msg_default_role_and_non_string_content():
    out = AssistantController._normalize_assistant_msg({})
    assert out["role"] == "assistant"
    assert out["content"] == ""

    out2 = AssistantController._normalize_assistant_msg({"role": "assistant", "content": None})
    assert out2["content"] == ""


def test_normalize_assistant_msg_coerces_model_dump_tool_calls():
    class FakeCall:
        def model_dump(self):
            return {"function": {"name": "y"}}
    msg = {"role": "assistant", "content": "", "tool_calls": [FakeCall()]}
    out = AssistantController._normalize_assistant_msg(msg)
    assert out["tool_calls"] == [{"function": {"name": "y"}}]


def test_normalize_assistant_msg_omits_empty_or_missing_tool_calls():
    out_empty = AssistantController._normalize_assistant_msg(
        {"role": "assistant", "content": "", "tool_calls": []},
    )
    assert "tool_calls" not in out_empty

    out_missing = AssistantController._normalize_assistant_msg({"role": "assistant", "content": ""})
    assert "tool_calls" not in out_missing


# --- submit_prompt system prompt augmentation ---


def test_submit_prompt_augments_system_prompt_with_existing_pets():
    backend = FakeBackend([
        _response_with_calls([_tool_call("finalize_plan", {"summary": "ok"})]),
    ])
    controller = AssistantController(backend=backend)
    controller.submit_prompt("hello", existing_pet_names=["Rex", "Felix"])

    system_msg = backend.sent_messages[0][0]
    assert system_msg["role"] == "system"
    assert "Rex" in system_msg["content"]
    assert "Felix" in system_msg["content"]
    assert "Existing pets in the tracker" in system_msg["content"]


def test_submit_prompt_no_augment_when_no_existing_pets():
    backend = FakeBackend([
        _response_with_calls([_tool_call("finalize_plan", {"summary": "ok"})]),
    ])
    controller = AssistantController(backend=backend)
    controller.submit_prompt("hello")

    system_msg = backend.sent_messages[0][0]
    assert "Existing pets in the tracker" not in system_msg["content"]
