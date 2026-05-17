# M1 Backend Contract

Legacy note: this file documents the original M1 singleton-session contract. The
current multi-conversation MVP contract is documented in
[`M3_API_CONTRACT.md`](M3_API_CONTRACT.md).

This is the intended boundary between the future FastAPI/UI layer and the M1 backend loop.

The UI should call backend functions or HTTP endpoints that map to these events. It should not duplicate quiz generation, grading, review scheduling, or SQLite persistence.

## Start Topic

Request:

```json
{
  "topic": "photosynthesis",
  "quiz_count": 3
}
```

Backend call:

```python
overview, questions = loop.start_topic(topic, quiz_count)
```

Response shape:

```json
{
  "overview": {
    "points": [
      "Label: description of key fact",
      "Another key fact"
    ]
  },
  "questions": [
    {
      "item_id": 1,
      "is_review": false,
      "question": "What is the primary energy source used by plants?",
      "choices": [
        "Chemical energy stored in glucose",
        "Heat energy from the environment",
        "Light energy from the sun",
        "Energy stored in water molecules"
      ]
    }
  ]
}
```

Do not send `answer_index` to the UI before the user answers.

## Answer Question

Request:

```json
{
  "item_id": 1,
  "choice_index": 2
}
```

Backend call:

```python
result = loop.answer_item(item_id, choice_index)
```

Response shape:

```json
{
  "item_id": 1,
  "is_correct": true,
  "correct_index": 2,
  "rationale": "Photosynthesis converts light energy into chemical energy."
}
```

Wrong answers are promoted into the review queue. Correct answers reduce priority and set a longer cooldown.

## Advance Turn

Request:

```json
{}
```

Backend call:

```python
review = loop.next_turn()
```

Response shape when no review is due:

```json
{
  "review": null
}
```

Response shape when a review question is due:

```json
{
  "review": {
    "item_id": 1,
    "is_review": true,
    "question": "What is the primary energy source used by plants?",
    "choices": [
      "Chemical energy stored in glucose",
      "Heat energy from the environment",
      "Light energy from the sun",
      "Energy stored in water molecules"
    ]
  }
}
```

## Error Cases

- Empty topic: `ValueError("topic cannot be empty")`
- Invalid `choice_index`: `ValueError("choice_index must be 0-3")`
- Unknown `item_id`: `ValueError("quiz item not found: <id>")`
- Invalid review interval: `ValueError("review_every must be at least 1")`
