import re
import textwrap

from balrog.agents.base import BaseAgent

# Answers that mean "keep the current plan".
KEEP_PLAN = {"", "unchanged", "no change", "no changes", "no changes to the plan", "same", "same plan", "none"}


class CustomAgent(BaseAgent):
    """An agent that generates actions and plans based on observations with a customizable approach."""

    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.plan = None

    def reset(self):
        # Called at the start of every episode: don't carry the old plan into a new mission.
        super().reset()
        self.plan = None

    def act(self, obs, prev_action=None):
        """Generate the next action and update the plan based on the current observation.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            LLMResponse: The response containing the updated plan and selected action.
        """
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)

        plan_text = f"Current Plan:\n{self.plan}\n" if self.plan else "You have no plan yet.\n"

        planning_instructions = textwrap.dedent("""
            First compare the current observation with your plan.
            Keep the plan if it still makes sense. Change it only if the observation shows
            something new that matters for your goal, or if your last actions did not help.

            Answer in exactly this format, with nothing after the ACTION line:
            THINK: <one sentence: what does the observation tell you about your goal?>
            PLAN: <at most 3 short steps, or "unchanged">
            ACTION: <exactly one valid action or command, with no explanation>
        """).strip()

        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + plan_text + "\n" + planning_instructions

        response = self.client.generate(messages)
        completion = response.completion or ""

        # Extract the thought, plan and action from the LLM's response
        think, plan, action = self._extract(completion)

        # Update the internal plan only if the model actually wrote a new one
        if plan is not None and plan.lower().rstrip(".") not in KEEP_PLAN:
            self.plan = plan

        # Keep THINK and PLAN in the reasoning field (it is saved in the trajectory CSV),
        # and the action in response.completion
        reasoning = f"THINK: {think or ''}\nPLAN: {self.plan or ''}"
        return response._replace(reasoning=reasoning, completion=action)

    @staticmethod
    def _clean(text):
        """First line only, no markdown/quotes, no trailing punctuation."""
        text = text.strip().splitlines()[0] if text.strip() else ""
        text = re.sub(r"[`*\"]", "", text)
        return text.strip().rstrip(".!?,;").strip()

    def _extract(self, response_text):
        """Extract (think, plan, action) from the LLM's response text.

        The action is never None: BALROG checks every *string* against the valid actions
        (an invalid one becomes the default action), but TextWorld accepts any value and
        crashes on None. If there is no ACTION line, fall back to the last non-empty line.
        """
        label = r"\**\s*{}\s*\**\s*:\s*\**"  # tolerates **PLAN:** and **PLAN**:
        think_match = re.search(label.format("THINK") + r"\s*(.*?)(?=\n\s*\**\s*(?:PLAN|ACTION)\b|\Z)",
                                response_text, re.IGNORECASE | re.DOTALL)
        plan_match = re.search(label.format("PLAN") + r"\s*(.*?)(?=\n\s*\**\s*ACTION\b|\Z)",
                               response_text, re.IGNORECASE | re.DOTALL)
        action_matches = re.findall(label.format("ACTION") + r"\s*(.*)", response_text, re.IGNORECASE)

        think = think_match.group(1).strip() if think_match else None
        plan = plan_match.group(1).strip() if plan_match else None

        if action_matches:
            action = self._clean(action_matches[-1])
        else:
            lines = [line for line in response_text.splitlines() if line.strip()]
            action = self._clean(lines[-1]) if lines else ""
        if not action:
            action = "none"  # not a valid action anywhere -> BALROG substitutes the default

        return think, plan, action