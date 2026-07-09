"""
Phase 2 — Build a labeled dataset of 200+ prompts across 3 complexity tiers.

Templates are combined with substitution lists to generate varied, realistic
examples without needing an external dataset. This is intentionally template-
based and hand-labeled-by-construction: every generated prompt's tier is known
because we generated it from a tier-specific template family.
"""
import random
from dataclasses import dataclass

random.seed(42)

TOPICS = [
    "the Q3 sales report", "this customer support ticket", "the attached invoice",
    "the user's resume", "this product review", "the meeting transcript",
    "the server log file", "this legal contract clause", "the patient intake form",
    "this GitHub issue", "the marketing email draft", "this API response payload",
    "the onboarding survey results", "this research abstract", "the shipping manifest",
    "this codebase's README", "the quarterly budget spreadsheet", "this support chat log",
]

ENTITIES = [
    "email address", "phone number", "order ID", "date", "total amount",
    "customer name", "shipping address", "product SKU", "invoice number", "due date",
]

# --- Tier 1: simple — reformatting, extraction, basic Q&A from provided context ---
TIER1_TEMPLATES = [
    "Extract the {entity} from {topic}.",
    "What is the {entity} mentioned in {topic}?",
    "Reformat {topic} as a bulleted list.",
    "Convert {topic} into plain text, removing all HTML tags.",
    "Pull out every {entity} found in {topic} and list them.",
    "Given the context, what does the {entity} say in {topic}?",
    "Capitalize the first letter of every sentence in {topic}.",
    "Return the {entity} from {topic} as a single value.",
    "Is there a {entity} present in {topic}? Answer yes or no.",
    "Copy the first paragraph of {topic} verbatim.",
]

# --- Tier 2: moderate — summarization, classification, structured analysis ---
TIER2_TEMPLATES = [
    "Summarize {topic} in three sentences.",
    "Classify {topic} as positive, negative, or neutral in tone.",
    "Organize the key points of {topic} into a JSON object with clear fields.",
    "Compare {topic} against last quarter's version and list the differences.",
    "Categorize each item in {topic} by department, and format as a table.",
    "Summarize {topic} and flag anything that looks like an error, using bullet points.",
    "Analyze the sentiment of {topic} and explain your reasoning in two sentences.",
    "Group the entries in {topic} by {entity} and return the counts as a table.",
    "Evaluate whether {topic} follows the standard format, and explain any deviations.",
    "Assess the urgency of {topic} on a 1-5 scale and justify the score briefly.",
]

# --- Tier 3: complex — multi-step reasoning, creative generation, nuanced judgment ---
TIER3_TEMPLATES = [
    "Analyze {topic} and {topic2}, weigh the trade-offs between them, and recommend "
    "a course of action, explaining your reasoning step by step.",
    "Write a short story inspired by the themes in {topic}, and explain the "
    "creative choices you made.",
    "Given {topic}, design a new process that would prevent the issues it describes, "
    "considering at least three constraints: cost, time, and team capacity.",
    "Critique the argument made in {topic}, identify any logical fallacies, and "
    "propose a stronger version of the argument.",
    "Brainstorm five original product ideas inspired by the gaps you find in {topic}, "
    "and rank them by feasibility, explaining your reasoning for each.",
    "Synthesize the findings across {topic} and {topic2} into a single coherent "
    "narrative, resolving any contradictions you find between them.",
    "Imagine you are advising a VP of Engineering: based on {topic}, justify why "
    "a particular architectural decision should or should not be made.",
    "Derive a root-cause explanation for the pattern shown in {topic}, considering "
    "multiple competing hypotheses before settling on the most likely one.",
    "Compose a nuanced response to {topic} that balances legal risk, customer "
    "satisfaction, and brand tone, and explain the trade-offs you considered.",
    "Prove, using the evidence in {topic}, why the proposed conclusion does or "
    "does not follow, addressing at least one counterargument.",
]


@dataclass
class Example:
    prompt: str
    context: str | None
    tier: int  # 1, 2, or 3


def _fill(template: str) -> str:
    topic = random.choice(TOPICS)
    topic2 = random.choice([t for t in TOPICS if t != topic])
    entity = random.choice(ENTITIES)
    return template.format(topic=topic, topic2=topic2, entity=entity)


def generate_dataset(per_tier: int = 80) -> list[Example]:
    """80 * 3 = 240 examples by default, comfortably above the 200+ target."""
    examples: list[Example] = []
    tier_templates = {1: TIER1_TEMPLATES, 2: TIER2_TEMPLATES, 3: TIER3_TEMPLATES}
    for tier, templates in tier_templates.items():
        for _ in range(per_tier):
            template = random.choice(templates)
            prompt = _fill(template)
            # ~40% of examples include a "context" blob, more common for tier 1/2
            has_context = random.random() < (0.55 if tier <= 2 else 0.25)
            context = (
                "Context: " + " ".join(random.choices(
                    ["Order #4471.", "Customer: Jane Doe.", "Total: $214.50.",
                     "Status: pending review.", "Submitted 2026-05-14.",
                     "Priority: high.", "Region: APAC."], k=random.randint(2, 5)
                ))
                if has_context else None
            )
            examples.append(Example(prompt=prompt, context=context, tier=tier))
    random.shuffle(examples)
    return examples


if __name__ == "__main__":
    data = generate_dataset()
    print(f"Generated {len(data)} labeled examples")
    for ex in data[:5]:
        print(f"[tier {ex.tier}] {ex.prompt}  | context={ex.context}")
