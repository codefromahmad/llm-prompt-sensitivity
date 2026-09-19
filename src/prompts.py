"""The preregistered, semantically equivalent prompt variants."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptVariant:
    identifier: str
    label: str
    instruction: str


PROMPT_VARIANTS: tuple[PromptVariant, ...] = (
    PromptVariant("P1", "Standard", "Choose the correct answer from the options below. Respond only with the letter A, B, C, D, or E."),
    PromptVariant("P2", "Concise", "Select the correct option. Answer only with A, B, C, D, or E."),
    PromptVariant("P3", "Polite", "Please choose the correct answer from the options provided. Reply only with A, B, C, D, or E."),
    PromptVariant("P4", "Alternative wording", "Determine which option correctly answers the question. Return only its letter: A, B, C, D, or E."),
    PromptVariant("P5", "Formal", "Identify the most appropriate answer among the five choices. Provide only the corresponding letter (A, B, C, D, or E)."),
)


def format_prompt(question: str, choices: dict[str, str], variant: PromptVariant) -> str:
    """Render question content identically for each variant, except instruction."""
    options = "\n".join(f"{letter}. {choices[letter]}" for letter in "ABCDE")
    return f"{variant.instruction}\n\nQuestion: {question}\n\nOptions:\n{options}"
