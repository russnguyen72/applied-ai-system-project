"""AI assistant controller: state machine, staged plans, and tool-call loop."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import date, time, timedelta
from enum import Enum
from typing import Callable

from llm_backend import LLMBackendError, OllamaBackend
from pawpal_system import Owner, Pet, Task

ProgressCallback = Callable[[str], None]
ThinkingCallback = Callable[[str], None]

MAX_TURNS = 8


class AssistantState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    READY_TO_REVIEW = "ready_to_review"
    APPLIED = "applied"
    ERROR = "error"


@dataclass
class StagedPet:
    name: str
    animal_type: str
    last_vet_visit: date | None = None
    selected: bool = True


@dataclass
class StagedTask:
    pet_name: str
    task_name: str
    description: str
    scheduled_time: time
    frequency_days: int
    start_in_days: int = 0
    selected: bool = True


@dataclass
class StagedPlan:
    pets: list[StagedPet] = field(default_factory=list)
    tasks: list[StagedTask] = field(default_factory=list)
    summary: str = ""


SYSTEM_PROMPT = """You are PawPal+'s scheduling assistant. The user describes pets and care routines in natural language, and you translate that into structured pet and task records by calling tools.

Rules:
- Always call `create_pet` before `create_task` for any pet that doesn't already exist.
- Reference existing pets by their exact name (case-sensitive).
- If a pet the user mentions already exists, do NOT create a duplicate — reuse it or call `ask_clarification` if it's ambiguous.
- If a required field (pet name, species, task time, or repeat frequency) is unclear, call `ask_clarification` with at most 4 focused questions, only one per required field, instead of guessing.
- You can do simple arithmetic for reminders. Example: 100 servings of food ÷ 4 servings/day (2 dogs × 2 meals) = 25 days. To remind when ~10 servings remain, schedule a reminder that starts in 22 days (when ~12 servings remain) and does not repeat (frequency_days=0).
- `scheduled_time` must be 24-hour HH:MM format (e.g., "07:30", "18:00").
- `frequency_days` is an integer: 0 for one-time tasks, 1 for daily, 7 for weekly, 30 for monthly, etc.
- `start_in_days` offsets when the first occurrence is due, counting from today. 0 means "today." Use this for reminders that shouldn't trigger until later.
- When you've staged everything the user asked for, call `finalize_plan` with a one-sentence summary.

Stay concise. Prefer calling tools over chatting."""


TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "create_pet",
            "description": "Stage a new pet to be added to the tracker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Pet's name."},
                    "animal_type": {
                        "type": "string",
                        "description": "Species of the pet, e.g. 'dog', 'cat', 'other'.",
                    },
                    "last_vet_visit": {
                        "type": "string",
                        "description": "Optional ISO date (YYYY-MM-DD) of the most recent vet visit.",
                    },
                },
                "required": ["name", "animal_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_task",
            "description": "Stage a care task for a pet. The pet must already exist or have been staged via create_pet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pet_name": {
                        "type": "string",
                        "description": "Exact name of the pet this task belongs to.",
                    },
                    "task_name": {"type": "string", "description": "Short task title, e.g. 'Morning Feed'."},
                    "description": {"type": "string", "description": "One-sentence description."},
                    "scheduled_time": {
                        "type": "string",
                        "description": "Time of day in 24-hour HH:MM format.",
                    },
                    "frequency_days": {
                        "type": "integer",
                        "description": "Repeat interval in days. 0 = one-time, 1 = daily, 7 = weekly.",
                    },
                    "start_in_days": {
                        "type": "integer",
                        "description": "Optional. Days from today before the first occurrence is due. Default 0.",
                    },
                },
                "required": ["pet_name", "task_name", "description", "scheduled_time", "frequency_days"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_clarification",
            "description": "Ask the user a single focused question when required information is missing or ambiguous.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "One clear question for the user."},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_plan",
            "description": "Call this when all pets and tasks have been staged and the plan is ready for user review.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "One-sentence recap of what was staged.",
                    },
                },
                "required": ["summary"],
            },
        },
    },
]


def _parse_time(s: str) -> time:
    return time.fromisoformat(s)


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


class AssistantController:
    def __init__(self, backend: OllamaBackend | None = None) -> None:
        self.backend = backend if backend is not None else OllamaBackend()
        self.state: AssistantState = AssistantState.IDLE
        self.messages: list[dict] = []
        self.staged_plan: StagedPlan = StagedPlan()
        self.pending_question: str | None = None
        self.error_message: str | None = None
        self.existing_pet_names: list[str] = []
        self.thinking_enabled: bool = False
        self.thinking_log: list[str] = []

    # --- Public API ---

    def submit_prompt(
        self,
        text: str,
        existing_pet_names: list[str] | None = None,
        progress_callback: ProgressCallback | None = None,
        thinking_callback: ThinkingCallback | None = None,
        thinking_enabled: bool = False,
    ) -> None:
        """Starts a new planning session from a user prompt."""
        self.reset()
        self.existing_pet_names = list(existing_pet_names or [])
        self.thinking_enabled = thinking_enabled
        system = SYSTEM_PROMPT
        if self.existing_pet_names:
            system += (
                "\n\nExisting pets in the tracker (do not re-create): "
                + ", ".join(self.existing_pet_names)
            )
        self.messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ]
        self.state = AssistantState.PLANNING
        self._emit("Reading your request", progress_callback)
        self._run_turn_loop(progress_callback, thinking_callback)

    def submit_clarification(
        self,
        text: str,
        progress_callback: ProgressCallback | None = None,
        thinking_callback: ThinkingCallback | None = None,
        thinking_enabled: bool | None = None,
    ) -> None:
        """Resumes planning after the user answers a clarifying question."""
        if self.state != AssistantState.AWAITING_CLARIFICATION:
            return
        if thinking_enabled is not None:
            self.thinking_enabled = thinking_enabled
        self.messages.append({"role": "user", "content": text})
        self.pending_question = None
        self.state = AssistantState.PLANNING
        self._emit("Processing your reply", progress_callback)
        self._run_turn_loop(progress_callback, thinking_callback)

    @staticmethod
    def _emit(label: str, cb: ProgressCallback | None) -> None:
        if cb is None:
            return
        try:
            cb(label)
        except Exception:
            pass

    def apply_to_owner(self, owner: Owner) -> None:
        """Commits the selected items from the staged plan to the real owner."""
        if self.state != AssistantState.READY_TO_REVIEW:
            return

        today = date.today()
        existing_names = {p.name for p in owner.pets}

        name_to_pet: dict[str, Pet] = {p.name: p for p in owner.pets}
        for staged_pet in self.staged_plan.pets:
            if not staged_pet.selected:
                continue
            if staged_pet.name in existing_names:
                continue
            new_pet = Pet(
                id=str(uuid.uuid4()),
                name=staged_pet.name,
                animal_type=staged_pet.animal_type,
                last_vet_visit=staged_pet.last_vet_visit,
            )
            owner.add_pet(new_pet)
            name_to_pet[new_pet.name] = new_pet
            existing_names.add(new_pet.name)

        for staged_task in self.staged_plan.tasks:
            if not staged_task.selected:
                continue
            pet = name_to_pet.get(staged_task.pet_name)
            if pet is None:
                continue
            pet.add_task(Task(
                id=str(uuid.uuid4()),
                name=staged_task.task_name,
                description=staged_task.description,
                scheduled_time=staged_task.scheduled_time,
                frequency_days=staged_task.frequency_days,
                next_due_date=today + timedelta(days=max(0, staged_task.start_in_days)),
            ))

        self.state = AssistantState.APPLIED

    def discard(self) -> None:
        """Throws away the staged plan and returns to idle."""
        self.staged_plan = StagedPlan()
        self.pending_question = None
        self.error_message = None
        self.messages = []
        self.state = AssistantState.IDLE

    def reset(self) -> None:
        """Clears all state, returning to a fresh IDLE."""
        self.staged_plan = StagedPlan()
        self.pending_question = None
        self.error_message = None
        self.messages = []
        self.thinking_log = []
        self.state = AssistantState.IDLE

    # --- Turn loop ---

    def _run_turn_loop(
        self,
        progress_cb: ProgressCallback | None = None,
        thinking_cb: ThinkingCallback | None = None,
    ) -> None:
        def _on_chunk(chunk: dict) -> None:
            if chunk.get("type") == "thinking":
                text = chunk.get("text", "")
                if text:
                    self.thinking_log.append(text)
                    if thinking_cb is not None:
                        try:
                            thinking_cb(text)
                        except Exception:
                            pass

        try:
            for _ in range(MAX_TURNS):
                self._emit("Thinking", progress_cb)
                response = self.backend.chat_stream(
                    self.messages,
                    TOOL_SCHEMAS,
                    think=self.thinking_enabled,
                    on_chunk=_on_chunk,
                )
                msg = self._extract_message(response)
                self.messages.append(self._normalize_assistant_msg(msg))

                tool_calls = self._extract_tool_calls(msg)
                if not tool_calls:
                    # No tool call — treat as implicit finish if we have content, else error.
                    if self.staged_plan.pets or self.staged_plan.tasks:
                        self.staged_plan.summary = self._extract_text(msg) or "Plan ready for review."
                        self.state = AssistantState.READY_TO_REVIEW
                        self._emit("Plan ready for review", progress_cb)
                    else:
                        self.error_message = (
                            "The model responded without calling any tool. "
                            "Try rephrasing your request with more detail."
                        )
                        self.state = AssistantState.ERROR
                        self._emit(f"Error: {self.error_message}", progress_cb)
                    return

                terminated = False
                for call in tool_calls:
                    name, args = self._extract_name_and_args(call)
                    self._emit(self._label_for_tool_call(name, args), progress_cb)
                    result_str = self._handle_tool_call(name, args)
                    self.messages.append({
                        "role": "tool",
                        "name": name,
                        "content": result_str,
                    })
                    if name == "ask_clarification":
                        self.state = AssistantState.AWAITING_CLARIFICATION
                        self._emit("Waiting for your reply", progress_cb)
                        terminated = True
                        break
                    if name == "finalize_plan":
                        self.state = AssistantState.READY_TO_REVIEW
                        self._emit("Plan ready for review", progress_cb)
                        terminated = True
                        break
                if terminated:
                    return

            # Exhausted turn budget.
            self.error_message = (
                f"The assistant ran {MAX_TURNS} turns without finalizing a plan. "
                "Try a simpler prompt or break the request into parts."
            )
            self.state = AssistantState.ERROR
            self._emit("Error: turn limit reached", progress_cb)
        except LLMBackendError as e:
            self.error_message = str(e)
            self.state = AssistantState.ERROR
            self._emit(f"Error: {self.error_message}", progress_cb)
        except Exception as e:
            self.error_message = f"Unexpected assistant error: {e}"
            self.state = AssistantState.ERROR
            self._emit(f"Error: {self.error_message}", progress_cb)

    @staticmethod
    def _label_for_tool_call(name: str, args: dict) -> str:
        if name == "create_pet":
            pet_name = args.get("name") or "pet"
            return f"Adding pet: {pet_name}"
        if name == "create_task":
            task_name = args.get("task_name") or "task"
            pet_name = args.get("pet_name") or "pet"
            return f"Scheduling {task_name} for {pet_name}"
        if name == "ask_clarification":
            return "Preparing a question"
        if name == "finalize_plan":
            return "Finalizing plan"
        return f"Processing {name}"

    # --- Tool dispatch ---

    def _handle_tool_call(self, name: str, args: dict) -> str:
        try:
            if name == "create_pet":
                return self._tool_create_pet(args)
            if name == "create_task":
                return self._tool_create_task(args)
            if name == "ask_clarification":
                return self._tool_ask_clarification(args)
            if name == "finalize_plan":
                return self._tool_finalize_plan(args)
            return json.dumps({"ok": False, "error": f"unknown tool '{name}'"})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def _tool_create_pet(self, args: dict) -> str:
        name = args.get("name")
        animal_type = args.get("animal_type")
        if not isinstance(name, str) or not name.strip():
            return json.dumps({"ok": False, "error": "'name' is required and must be a non-empty string"})
        if not isinstance(animal_type, str) or not animal_type.strip():
            return json.dumps({"ok": False, "error": "'animal_type' is required and must be a non-empty string"})

        last_vet_raw = args.get("last_vet_visit")
        last_vet: date | None = None
        if last_vet_raw:
            try:
                last_vet = _parse_date(last_vet_raw)
            except ValueError:
                return json.dumps({"ok": False, "error": "'last_vet_visit' must be YYYY-MM-DD"})

        if name in self.existing_pet_names:
            return json.dumps({
                "ok": False,
                "error": f"A pet named '{name}' already exists in the tracker. Reference it by name instead.",
            })
        if any(p.name == name for p in self.staged_plan.pets):
            return json.dumps({
                "ok": False,
                "error": f"A pet named '{name}' is already staged.",
            })

        self.staged_plan.pets.append(StagedPet(
            name=name.strip(),
            animal_type=animal_type.strip().lower(),
            last_vet_visit=last_vet,
        ))
        return json.dumps({"ok": True, "staged_pet": name})

    def _tool_create_task(self, args: dict) -> str:
        required = ["pet_name", "task_name", "description", "scheduled_time", "frequency_days"]
        for key in required:
            if key not in args:
                return json.dumps({"ok": False, "error": f"missing required field '{key}'"})

        pet_name = args["pet_name"]
        if not isinstance(pet_name, str) or not pet_name.strip():
            return json.dumps({"ok": False, "error": "'pet_name' must be a non-empty string"})

        known_names = set(self.existing_pet_names) | {p.name for p in self.staged_plan.pets}
        if pet_name not in known_names:
            return json.dumps({
                "ok": False,
                "error": (
                    f"Pet '{pet_name}' does not exist. Call create_pet first, "
                    f"or use one of: {sorted(known_names) or '(none)'}"
                ),
            })

        task_name = args["task_name"]
        description = args["description"]
        if not isinstance(task_name, str) or not task_name.strip():
            return json.dumps({"ok": False, "error": "'task_name' must be a non-empty string"})
        if not isinstance(description, str):
            return json.dumps({"ok": False, "error": "'description' must be a string"})

        scheduled_time_raw = args["scheduled_time"]
        try:
            scheduled_time = _parse_time(str(scheduled_time_raw))
        except ValueError:
            return json.dumps({
                "ok": False,
                "error": f"'scheduled_time' must be HH:MM (24-hour), got '{scheduled_time_raw}'",
            })

        frequency_days = args["frequency_days"]
        if isinstance(frequency_days, bool) or not isinstance(frequency_days, int):
            try:
                frequency_days = int(frequency_days)
            except (ValueError, TypeError):
                return json.dumps({"ok": False, "error": "'frequency_days' must be a non-negative integer"})
        if frequency_days < 0:
            return json.dumps({"ok": False, "error": "'frequency_days' must be >= 0"})

        start_in_days = args.get("start_in_days", 0)
        if isinstance(start_in_days, bool) or not isinstance(start_in_days, int):
            try:
                start_in_days = int(start_in_days)
            except (ValueError, TypeError):
                return json.dumps({"ok": False, "error": "'start_in_days' must be a non-negative integer"})
        if start_in_days < 0:
            return json.dumps({"ok": False, "error": "'start_in_days' must be >= 0"})

        self.staged_plan.tasks.append(StagedTask(
            pet_name=pet_name,
            task_name=task_name.strip(),
            description=description,
            scheduled_time=scheduled_time,
            frequency_days=frequency_days,
            start_in_days=start_in_days,
        ))
        return json.dumps({"ok": True, "staged_task": task_name})

    def _tool_ask_clarification(self, args: dict) -> str:
        question = args.get("question")
        if not isinstance(question, str) or not question.strip():
            return json.dumps({"ok": False, "error": "'question' is required"})
        self.pending_question = question.strip()
        return json.dumps({"ok": True, "awaiting_user": True})

    def _tool_finalize_plan(self, args: dict) -> str:
        summary = args.get("summary", "")
        if not isinstance(summary, str):
            summary = ""
        self.staged_plan.summary = summary.strip() or "Plan ready for review."
        return json.dumps({"ok": True, "finalized": True})

    # --- Response parsing (handles both Ollama dict and object shapes) ---

    @staticmethod
    def _extract_message(response: dict) -> dict:
        msg = response.get("message") if isinstance(response, dict) else None
        if msg is None:
            return {}
        if hasattr(msg, "model_dump"):
            return msg.model_dump()
        if isinstance(msg, dict):
            return msg
        return dict(msg)

    @staticmethod
    def _extract_tool_calls(msg: dict) -> list:
        calls = msg.get("tool_calls") or []
        return list(calls)

    @staticmethod
    def _extract_text(msg: dict) -> str:
        content = msg.get("content")
        if isinstance(content, str):
            return content
        return ""

    @staticmethod
    def _extract_name_and_args(call) -> tuple[str, dict]:
        if hasattr(call, "model_dump"):
            call = call.model_dump()
        if not isinstance(call, dict):
            call = dict(call)
        fn = call.get("function") or {}
        if hasattr(fn, "model_dump"):
            fn = fn.model_dump()
        name = fn.get("name") or call.get("name") or ""
        raw_args = fn.get("arguments") if isinstance(fn, dict) else None
        if raw_args is None:
            raw_args = call.get("arguments")
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError:
                raw_args = {}
        if not isinstance(raw_args, dict):
            raw_args = {}
        return name, raw_args

    @staticmethod
    def _normalize_assistant_msg(msg: dict) -> dict:
        """Coerces the assistant's message into the dict shape Ollama expects on echo-back."""
        normalized: dict = {"role": msg.get("role", "assistant")}
        content = msg.get("content")
        normalized["content"] = content if isinstance(content, str) else ""
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            coerced = []
            for call in tool_calls:
                if hasattr(call, "model_dump"):
                    call = call.model_dump()
                coerced.append(call)
            normalized["tool_calls"] = coerced
        return normalized
