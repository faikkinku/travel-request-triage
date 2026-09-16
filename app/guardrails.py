"""The independent guardrail check — plain Python, no model involved.

This is the most important file in the project. Everything else asks the model to
behave; this decides. The model is told to escalate certain situations, and
separately this code checks the client's own words. If either says escalate, the
case escalates.

That difference matters: a prompt is a request, and a confused or manipulated model
can ignore it. A function that runs on every request regardless of what the model
returned is a mechanism.

Design bias: over-triggering here costs a manager a few minutes of review.
Under-triggering costs a client who needed help right away. Ambiguity therefore
resolves toward escalation.
"""

import re

from app.schemas import TriageResult, Urgency

# Each entry: (label, regex). Patterns are lowercase; the message is lowercased first.
OVERRIDE_TRIGGERS: list[tuple[str, str]] = [
    # Legal and financial escalation — once a client says these words, the agency's
    # exposure changes regardless of how routine the underlying request looked.
    (
        "legal or chargeback threat",
        r"charge ?back|disput\w* (this|the|that) (charge|payment) with my bank|"
        r"(contact|call|involve|talk to|already spoke with) (my |a )?(lawyer|attorney)|"
        r"sue (you|the company|your agency|this agency)|legal action|"
        r"better business bureau|\bbbb\b (complaint|report)|small claims",
    ),
    (
        "public escalation threat",
        r"(post|going to post|will post) (this|it) (on|to) (social media|twitter|x|instagram|tiktok|facebook)|"
        r"make (this|it) go viral|(review|post).{0,15}(everywhere|every platform)",
    ),
    # Stranded or already at risk — these mean someone is affected right now, not on
    # a hypothetical future trip.
    (
        "denied boarding or stranded",
        r"denied boarding|stuck at the airport|\bstranded\b|"
        r"bumped from (my|the|our) flight|missed (my|our) (flight|connection) because",
    ),
    (
        "same-day or imminent departure",
        r"(flight|trip|departure) (is|leaves|departs) (today|tonight|tomorrow|in \d+ ?hours?)|"
        r"leaving (today|tonight|tomorrow)|(departs?|leaves?) in (the next )?\d+ ?hours?",
    ),
    # Medical and family emergencies change what "routine" means entirely.
    (
        "medical emergency or bereavement",
        r"medical emergency|hospitaliz\w+|in (the |an )?icu\b|"
        r"(a |)death in the family|passed away|\bfuneral\b|bereavement",
    ),
    (
        "unaccompanied minor",
        r"(child|kid|daughter|son|minor).{0,25}(travel(l)?ing|flying|going) alone|"
        r"unaccompanied minor",
    ),
    # Documents close to travel date — the agency cannot make visa determinations,
    # but a denial or expiry this close to departure needs a human today.
    (
        "visa or passport denial near travel",
        r"(visa|passport).{0,60}(denied|rejected)|(denied|rejected).{0,20}(visa|passport)|"
        r"passport.{0,20}expir\w+.{0,30}(before|during|within|by the time)",
    ),
    # A dollar figure this size moving without a human is exactly the failure mode
    # the manager-review threshold exists to prevent.
    (
        "large refund or compensation amount",
        r"\$\s?([2-9],?[0-9]{3}|[1-9][0-9]{4,})\b",
    ),
]

# Deliberately narrow. A wide negation rule would silently suppress a real trigger,
# which is the one failure this file exists to prevent.
NEGATION_CUES = (
    "no ", "not ", "never ", "without ", "denies ", "denied ", "don't have ",
    "dont have ", "doesn't have ", "haven't had ", "havent had ", "won't need ",
)

# If any of these follow a negated match, the negation is not trusted — the client
# is drawing a contrast, and the second half may still matter.
CONTRAST_CUES = (" but ", " however ", " although ", " though ", " except ")

COMPILED = [(label, re.compile(pattern)) for label, pattern in OVERRIDE_TRIGGERS]


def _is_negated(text: str, start: int) -> bool:
    """True when a trigger phrase is directly denied by the client.

    Only the ~28 characters immediately before the match are considered, so
    "no chargeback needed" is negated while "no delay, but I will file a
    chargeback" is not. Anything less obvious is treated as a real trigger on
    purpose.
    """
    window = text[max(0, start - 28) : start]
    if not any(cue in window for cue in NEGATION_CUES):
        return False
    return not any(cue in window for cue in CONTRAST_CUES)


def find_overrides(message: str) -> list[str]:
    """Return the labels of every override trigger found in the client's own words."""
    text = " " + re.sub(r"\s+", " ", message.lower().replace("’", "'")) + " "
    found: list[str] = []
    for label, pattern in COMPILED:
        for match in pattern.finditer(text):
            if _is_negated(text, match.start()):
                continue
            if label not in found:
                found.append(label)
            break
    return found


INJECTION_PATTERNS = [
    r"ignore .{0,30}(previous|prior|above|earlier) .{0,20}(instruction|rule|prompt)",
    r"disregard .{0,30}(instruction|rule|guideline|policy)",
    r"(you are|act as|pretend to be|from now on you) .{0,40}(different|new|no longer)",
    r"(skip|bypass|disable|turn off) .{0,30}(check|rule|policy|approval|guardrail|review)",
    r"(reveal|show|print|repeat) .{0,30}(system prompt|your instructions|the prompt)",
    r"(mark|set|flag) (this|it) .{0,15}(routine|low priority|no escalation)",
    r"do not escalate|don'?t escalate|no need (for|to) (a |)manager",
    # Impersonating an operator channel.
    r"^\s*(system|admin|developer)\s*[:>]",
    r"(approve|refund|waive|escalate)[\"']?\s*[=:]\s*[\"']?(true|false|yes|no|\$?\d)",
    # Instructions hidden in markup a client would never type.
    r"<!--.{0,80}(assistant|system|instruction|escalate|approve)",
    r"<\s*(system|instruction|prompt)\s*>",
    r"(give|issue|process) (me |the client |)a (full |)refund (automatically|right away|without (a |)review)",
]
COMPILED_INJECTION = [re.compile(p) for p in INJECTION_PATTERNS]


def find_injection_attempts(message: str) -> list[str]:
    """Spot text trying to rewrite the rules rather than describe a request.

    A client's message is data, not instruction. We cannot stop the model from
    reading such text, but we can notice it, record it, and escalate the case to a
    human.
    """
    text = re.sub(r"\s+", " ", message.lower().replace("’", "'"))
    return [p.pattern for p in COMPILED_INJECTION if p.search(text)]


LOW_CONFIDENCE_THRESHOLD = 0.5


def apply_guardrails(result: TriageResult, message: str) -> tuple[TriageResult, list[str]]:
    """Correct the model's answer using checks the model had no part in.

    Returns the corrected result and the triggers our own code found. The model's
    answer is never trusted to lower urgency below what these rules require, but it
    is also never overruled downward — a model that escalates on something this
    list does not cover keeps its escalation.
    """
    triggers = find_overrides(message)
    injections = find_injection_attempts(message)

    corrected = result.model_copy(deep=True)
    reasons: list[str] = []

    if triggers:
        reasons.append("override triggers detected by the agency's own check: " + "; ".join(triggers))
    if injections:
        reasons.append("the message tried to modify the assistant's instructions")
    if corrected.confidence < LOW_CONFIDENCE_THRESHOLD:
        reasons.append(f"confidence {corrected.confidence:.2f} is below the {LOW_CONFIDENCE_THRESHOLD} threshold")

    if reasons:
        corrected.escalate = True
        existing = (corrected.escalate_reason or "").strip()
        forced = "Escalated by the agency's independent check: " + "; ".join(reasons) + "."
        corrected.escalate_reason = f"{existing} {forced}".strip() if existing else forced

    # Triggers also set a floor under urgency. The model does not get to call a
    # stranded, denied-boarding client "routine", whatever it was asked to do.
    if triggers and corrected.urgency in (Urgency.ROUTINE, Urgency.SOON):
        corrected.urgency = Urgency.IMMEDIATE
    if injections and corrected.urgency == Urgency.ROUTINE:
        corrected.urgency = Urgency.URGENT

    return corrected, triggers
