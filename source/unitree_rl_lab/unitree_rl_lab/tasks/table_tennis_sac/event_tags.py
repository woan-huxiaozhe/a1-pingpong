from __future__ import annotations

EVENT_TAGS: tuple[str, ...] = ("near_miss", "hit", "bad_hit", "return", "valid_return", "miss")
EVENT_TO_BIT = {name: 1 << idx for idx, name in enumerate(EVENT_TAGS)}
BIT_TO_EVENT = {bit: name for name, bit in EVENT_TO_BIT.items()}


def encode_events(events: list[str] | tuple[str, ...] | set[str]) -> int:
    mask = 0
    for event in events:
        mask |= EVENT_TO_BIT[event]
    return mask


def decode_events(mask: int) -> list[str]:
    return [name for name, bit in EVENT_TO_BIT.items() if mask & bit]
