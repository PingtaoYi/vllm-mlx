# SPDX-License-Identifier: Apache-2.0
"""
Base parser for models using <think>...</think> tags for reasoning.

This module provides BaseThinkingReasoningParser, a concrete implementation
for extracting reasoning content from models that use thinking tags.

Supports three scenarios:
1. Both tags in output: <think>reasoning</think>content
2. Only closing tag (think injected in prompt): reasoning</think>content
3. No tags: pure content

Performance: The streaming parser uses a simple state machine to track the
current phase (pre-think / thinking / content). Each token is classified in
O(1) by checking only the delta text — the accumulated output is never
rescanned. This keeps per-token overhead constant regardless of output length.
"""

from abc import abstractmethod

from .base import DeltaMessage, ReasoningParser


class BaseThinkingReasoningParser(ReasoningParser):
    """
    Base parser for models using <think>...</think> style tags.

    This parser handles the common pattern where reasoning content is wrapped
    in special tags. Subclasses define the specific start and end tokens.

    Supports "implicit reasoning mode" where <think> is injected in the prompt
    and only </think> appears in the model output. This is common with AI agents
    like OpenCode that force models to reason by injecting thinking tags.

    The streaming parser uses a state machine with three phases:

        pre_think -> thinking -> content

    Transitions happen when start/end tokens are detected in the delta text.
    No accumulated text scanning is performed — each token is O(1).
    """

    @property
    @abstractmethod
    def start_token(self) -> str:
        """The token/tag that starts reasoning content (e.g., '<think>')."""

    @property
    @abstractmethod
    def end_token(self) -> str:
        """The token/tag that ends reasoning content (e.g., '</think>')."""

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        # Streaming state — reset per request via reset_state()
        self._phase: str = "pre_think"  # "pre_think" | "thinking" | "content"

    def reset_state(self):
        """Reset state machine for a new streaming request."""
        self._phase = "pre_think"

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from complete output.

        Handles three cases:
        1. Both tags present: <think>reasoning</think>content
        2. Only closing tag: reasoning</think>content (think in prompt)
        3. No tags: pure content

        Args:
            model_output: Complete model output text.

        Returns:
            (reasoning, content) tuple. Either may be None.
        """
        text = model_output

        # Case 1: Both tags present (normal case)
        if self.start_token in text and self.end_token in text:
            _, _, after_start = text.partition(self.start_token)
            reasoning, _, content = after_start.partition(self.end_token)
            return reasoning.strip() or None, content.strip() or None

        # Case 2: Only closing tag (think was injected in prompt)
        if self.end_token in text:
            reasoning, _, content = text.partition(self.end_token)
            return reasoning.strip() or None, content.strip() or None

        # Case 3: Only start tag (incomplete reasoning, no end yet)
        if self.start_token in text:
            _, _, reasoning = text.partition(self.start_token)
            return reasoning.strip() or None, None

        # Case 4: No tags at all — pure content
        return None, model_output

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from a streaming delta using state-machine tracking.

        Instead of rescanning the full accumulated text on every token, this
        method tracks the current phase (pre_think / thinking / content) and
        only inspects the delta for tag transitions. This makes each call O(1)
        regardless of how much text has been generated.

        The method signature is kept compatible with the base class — previous_text
        and current_text are accepted but not used for phase detection (they remain
        available for subclasses that need them).

        Handles three scenarios:
        1. Explicit <think>...</think> in model output
        2. Implicit mode (<think> in prompt, only </think> in output)
        3. No tags at all (pure content after first token with no reasoning)

        Args:
            previous_text: Text accumulated before this delta (unused by state machine).
            current_text: Text including this delta (unused by state machine).
            delta_text: Just the new text in this chunk.

        Returns:
            DeltaMessage with reasoning and/or content, or None to skip.
        """
        if not delta_text:
            return None

        start_tok = self.start_token
        end_tok = self.end_token

        # ── Phase: pre_think ──────────────────────────────────────
        # Haven't seen any tags yet. Could be:
        # - About to see <think> (explicit reasoning)
        # - Already inside implicit reasoning (think was in prompt)
        # - No reasoning at all (pure content model)
        if self._phase == "pre_think":
            # Check for start tag in this delta
            if start_tok in delta_text:
                self._phase = "thinking"
                idx = delta_text.find(start_tok) + len(start_tok)
                after = delta_text[idx:]
                # Edge case: both tags in same delta
                if end_tok in after:
                    self._phase = "content"
                    eidx = after.find(end_tok)
                    reasoning = after[:eidx]
                    content = after[eidx + len(end_tok):]
                    return DeltaMessage(
                        reasoning=reasoning or None,
                        content=content or None,
                    )
                return DeltaMessage(reasoning=after) if after else None

            # Check for end tag (implicit mode — think was in prompt)
            if end_tok in delta_text:
                self._phase = "content"
                idx = delta_text.find(end_tok)
                reasoning = delta_text[:idx]
                content = delta_text[idx + len(end_tok):]
                return DeltaMessage(
                    reasoning=reasoning or None,
                    content=content or None,
                )

            # No tags — default to reasoning (implicit mode assumption).
            # If the model doesn't use thinking at all, the server's
            # non-parser path handles it. This path only activates when
            # a reasoning parser is explicitly configured.
            return DeltaMessage(reasoning=delta_text)

        # ── Phase: thinking ───────────────────────────────────────
        # Inside a reasoning block, waiting for end tag.
        if self._phase == "thinking":
            if end_tok in delta_text:
                self._phase = "content"
                idx = delta_text.find(end_tok)
                reasoning = delta_text[:idx]
                content = delta_text[idx + len(end_tok):]
                return DeltaMessage(
                    reasoning=reasoning or None,
                    content=content or None,
                )
            return DeltaMessage(reasoning=delta_text)

        # ── Phase: content ────────────────────────────────────────
        # Past the reasoning block — everything is content.
        return DeltaMessage(content=delta_text)
