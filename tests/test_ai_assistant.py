"""Tests for the AI assistant controller. Uses a FakeBackend — no network calls."""
from __future__ import annotations

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
