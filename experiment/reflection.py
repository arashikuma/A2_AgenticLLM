import re
import textwrap
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

from balrog.agents.base import BaseAgent

# Define a minimal Message schema to satisfy BALROG client's attribute expectations
@dataclass
class EvalMessage:
    role: str
    content: str
    image: Optional[Any] = None
    attachment: Optional[Any] = None  

# Semantic equivalents indicating no plan modification
KEEP_PLAN = {
    "", "unchanged", "no change", "no changes",
    "no changes to the plan", "same", "same plan", "none"
}

class ReflectionAgent(BaseAgent):
    """
    Self-Reflective Agent for Exercise 3.
    Implements a dual-role cognitive architecture:
      1. Evaluator (Critic): Diagnoses failed actions and formulates concise rules (lessons).
      2. Actor: Generates actions based on observations, current plans, and the episodic memory buffer.
    """

    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.plan = None
        self.last_obs_repr = None
        self.last_action = None
        self.consecutive_repeats = 0
        # Bounded episodic memory buffer (max 3 recent lessons) to prevent context drift
        self.reflection_memory = deque(maxlen=3)

    def reset(self):
        """Clear internal states across episodes to prevent memory leakage."""
        super().reset()
        self.plan = None
        self.last_obs_repr = None
        self.last_action = None
        self.consecutive_repeats = 0
        self.reflection_memory.clear()

    @staticmethod
    def _clean(text: str) -> str:
        """Strip markdown markers, quotes, and trailing punctuation."""
        if not text or not text.strip():
            return ""
        first_line = text.strip().splitlines()[0]
        return re.sub(r"[`*\"]", "", first_line).strip().rstrip(".!?,;").strip()

    @staticmethod
    def _format_obs(obs) -> str:
        """Extract textual representation of the observation, omitting large image arrays."""
        if isinstance(obs, dict):
            return str({k: v for k, v in obs.items() if k != "image"})
        return str(obs)

    def _extract_actor_output(self, response_text: str):
        """Parse THINK, PLAN, and ACTION components from the model's completion."""
        label_pattern = r"\**\s*{}\s*\**\s*:\s*\**"
        
        think_match = re.search(
            label_pattern.format("THINK") + r"\s*(.*?)(?=\n\s*\**\s*(?:PLAN|ACTION)\b|\Z)",
            response_text, re.IGNORECASE | re.DOTALL
        )
        plan_match = re.search(
            label_pattern.format("PLAN") + r"\s*(.*?)(?=\n\s*\**\s*ACTION\b|\Z)",
            response_text, re.IGNORECASE | re.DOTALL
        )
        action_matches = re.findall(
            label_pattern.format("ACTION") + r"\s*(.*)", response_text, re.IGNORECASE
        )

        think = think_match.group(1).strip() if think_match else None
        plan = plan_match.group(1).strip() if plan_match else None

        if action_matches:
            action = self._clean(action_matches[-1])
        else:
            lines = [line for line in response_text.splitlines() if line.strip()]
            action = self._clean(lines[-1]) if lines else "none"

        return think, plan, action or "none"

    def _evaluate_failure(self, current_obs_str: str, failed_action: str) -> str:
        """
        Role 1: Evaluator. 
        Triggered when stagnation is detected. Diagnoses why the action failed and returns a rule.
        """
        eval_prompt = textwrap.dedent(f"""
            You are an analytical evaluator monitoring an autonomous agent in a puzzle environment.
            The agent executed the action: "{failed_action}".
            Outcome: The observation did not change (e.g., hit a wall) or the action was invalid.
            Current Observation: {current_obs_str}

            Diagnose the mistake in 1 short sentence and provide 1 concrete lesson to avoid repeating it.
            Format strictly as:
            DIAGNOSIS: <why it failed>
            LESSON: <specific rule or alternative direction for the next step>
        """).strip()

        # Wrapped in EvalMessage to fulfill BALROG client's msg.content object interface
        eval_messages = [EvalMessage(role="user", content=eval_prompt)]

        eval_resp = self.client.generate(eval_messages)
        completion = eval_resp.completion or ""

        lesson_match = re.search(r"LESSON:\s*(.*)", completion, re.IGNORECASE)
        if lesson_match:
            return self._clean(lesson_match.group(1))
        return f"Action '{failed_action}' failed; try a different direction or object."

    def act(self, obs, prev_action=None):
        """
        Role 2: Actor. 
        Generates actions conditioned on observations, the current plan, and reflection memory.
        """
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)

        current_obs_str = self._format_obs(obs)

        # 1. Stagnation Detection Heuristics
        is_stuck = False
        if prev_action and self.last_obs_repr is not None:
            # Perceptual stagnation (hit a wall / invalid action)
            if current_obs_str == self.last_obs_repr:
                is_stuck = True
            # Consecutive looping behavior
            if prev_action == self.last_action:
                self.consecutive_repeats += 1
                if self.consecutive_repeats >= 2:
                    is_stuck = True
            else:
                self.consecutive_repeats = 0

        # 2. Trigger Evaluator Reflection
        if is_stuck and prev_action:
            new_lesson = self._evaluate_failure(current_obs_str, prev_action)
            self.reflection_memory.append(new_lesson)
            # Invalidate stale plan to force cognitive re-evaluation
            self.plan = None

        # 3. Dynamic Prompt Assembly
        memory_text = ""
        if self.reflection_memory:
            lessons_joined = "\n".join([f"- {l}" for l in self.reflection_memory])
            memory_text = f"Reflective Lessons (What failed previously):\n{lessons_joined}\n\n"

        plan_text = f"Current Plan:\n{self.plan}\n\n" if self.plan else "You have no plan yet.\n\n"

        instructions = textwrap.dedent("""
            Compare your observation with your plan and past reflective lessons.
            Do not repeat actions that previously failed. Keep your plan if still valid; otherwise revise it.

            Answer in exactly this format, with nothing after ACTION:
            THINK: <one sentence: what does the observation tell you regarding past lessons?>
            PLAN: <at most 3 short steps, or "unchanged">
            ACTION: <exactly one valid action or command, no explanations>
        """).strip()

        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            messages[-1].content += f"\n\n{memory_text}{plan_text}{instructions}"

        # 4. Generate Actor Output
        response = self.client.generate(messages)
        completion = response.completion or ""

        think, plan, action = self._extract_actor_output(completion)

        if plan is not None and plan.lower().rstrip(".") not in KEEP_PLAN:
            self.plan = plan

        # Update historical state trackers
        self.last_obs_repr = current_obs_str
        self.last_action = action

        # 5. Inject Reasoning Trace for Trajectory CSV
        reasoning_trace = (
            f"REFLECTIONS: {list(self.reflection_memory)}\n"
            f"THINK: {think or ''}\n"
            f"PLAN: {self.plan or ''}"
        )
        return response._replace(reasoning=reasoning_trace, completion=action)