from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json, math, torch, numpy as np
from collections import Counter
import torch.nn as nn
from huggingface_hub import hf_hub_download

HF_REPO_ID = "sato2ru/wordle-solver"

app = FastAPI(title="Wordle Solver API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load assets ───────────────────────────────────────────────────────────────
print("Loading model...")
config   = json.load(open(hf_hub_download(HF_REPO_ID, "config.json")))
ANSWERS  = json.load(open(hf_hub_download(HF_REPO_ID, "answers.json")))
ALLOWED  = json.load(open(hf_hub_download(HF_REPO_ID, "allowed.json")))
WORD2IDX = {w: i for i, w in enumerate(ALLOWED)}
LETTERS  = "abcdefghijklmnopqrstuvwxyz"
L2I      = {c: i for i, c in enumerate(LETTERS)}
INPUT_DIM   = config["input_dim"]
OUTPUT_DIM  = config["output_dim"]
OPENING     = config["opening_guess"]
WIN_PATTERN = (2, 2, 2, 2, 2)

# ── Model ─────────────────────────────────────────────────────────────────────
class WordleNet(nn.Module):
    def __init__(self):
        super().__init__()
        h = config["hidden"]
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h, h),         nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h, 256),       nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, OUTPUT_DIM)
        )
    def forward(self, x): return self.net(x)

model = WordleNet()
model.load_state_dict(
    torch.load(hf_hub_download(HF_REPO_ID, "model_weights.pt"), map_location="cpu")
)
model.eval()
print("Model loaded ✅")

# ── Core helpers ──────────────────────────────────────────────────────────────
def get_pattern(guess, answer):
    pattern = [0] * 5
    counts  = Counter(answer)
    for i in range(5):
        if guess[i] == answer[i]:
            pattern[i] = 2
            counts[guess[i]] -= 1
    for i in range(5):
        if pattern[i] == 0 and counts.get(guess[i], 0) > 0:
            pattern[i] = 1
            counts[guess[i]] -= 1
    return tuple(pattern)

def filter_words(words, guess, pattern):
    return [w for w in words if get_pattern(guess, w) == tuple(pattern)]

def entropy_score(guess, possible):
    buckets = Counter(get_pattern(guess, w) for w in possible)
    n = len(possible)
    return sum(-(c / n) * math.log2(c / n) for c in buckets.values())

def encode_board(history):
    vec = np.zeros(INPUT_DIM, dtype=np.float32)
    for word, pattern in history:
        for pos, (letter, state) in enumerate(zip(word, pattern)):
            vec[L2I[letter] * 15 + pos * 3 + state] = 1.0
    return vec

def is_consistent(word, history):
    for guess, pattern in history:
        # Collect green letters in this guess
        green_letters = {letter for letter, state in zip(guess, pattern) if state == 2}
        
        for pos, (letter, state) in enumerate(zip(guess, pattern)):
            if state == 2:
                if word[pos] != letter:
                    return False
            elif state == 1:
                if letter not in word or word[pos] == letter:
                    return False
            else:  # grey
                # Only exclude if this letter isn't green somewhere in THIS guess
                if letter not in green_letters and letter in word:
                    return False
    return True

# ── Suggestion logic ──────────────────────────────────────────────────────────
def model_suggest(history, possible):
    """Pick the best next guess given history (list of (word,pattern) tuples)
    and the current list of possible answers."""
    if not possible:        return None
    if len(possible) == 1:  return possible[0]
    if not history:         return OPENING

    already_guessed = {w for w, _ in history}

    # ── Late-game elimination mode (≤ 6 candidates left) ─────────────────────
    if len(possible) <= 6:
        # Which letters are still ambiguous across the remaining words?
        ambiguous = set()
        for pos in range(5):
            letters_at_pos = {w[pos] for w in possible}
            if len(letters_at_pos) > 1:
                ambiguous.update(letters_at_pos)

        best_word, best_score = None, -1
        for g in ALLOWED:
            if g in already_guessed:
                continue
            if not is_consistent(g, history):
                continue
            # Prefer non-answer words when multiple candidates remain
            # so we eliminate as many as possible in one shot
            if g in possible and len(possible) > 2:
                continue
            score    = len(set(g) & ambiguous) * 2 + entropy_score(g, possible)
            if score > best_score:
                best_score, best_word = score, g

        
        # Fall back to any remaining possible word
        if best_word is None:
            candidates = [w for w in possible if w not in already_guessed]
            if not candidates:
                candidates = possible
            best_word = max(candidates, key=lambda w: entropy_score(w, possible))

        return best_word

    # ── Normal model inference ────────────────────────────────────────────────
    state = torch.tensor(encode_board(history)).unsqueeze(0)
    with torch.no_grad():
        logits = model(state)[0]

    # Take top-50 from model, then filter for consistency + not already guessed
    top50 = [ALLOWED[i] for i in logits.topk(50).indices.tolist()]
    valid = [w for w in top50 if w not in already_guessed and is_consistent(w, history)]

    if not valid:
        # Fallback: best entropy among remaining possible words
        fallback = [w for w in possible if w not in already_guessed]
        return max(fallback, key=lambda w: entropy_score(w, possible)) if fallback else possible[0]

    return max(valid[:10], key=lambda w: entropy_score(w, possible))


def top_suggestions(history, possible, n=5):
    """Return top N suggestions with entropy scores."""
    if not possible: return []

    already_guessed = {w for w, _ in history}

    if not history:
        candidates = [OPENING] + [w for w in ALLOWED if w != OPENING][:30]
    else:
        state = torch.tensor(encode_board(history)).unsqueeze(0)
        with torch.no_grad():
            logits = model(state)[0]
        candidates = [ALLOWED[i] for i in logits.topk(50).indices.tolist()]

    # Filter: consistent with history, not already guessed
    candidates = [w for w in candidates
                  if w not in already_guessed and is_consistent(w, history)]

    possible_set = set(possible)
    scored = [
        {
            "word":        w,
            "entropy":     round(entropy_score(w, possible), 3),
            "is_possible": w in possible_set,
        }
        for w in candidates
    ]
    scored.sort(key=lambda x: (-x["entropy"], not x["is_possible"]))
    return scored[:n]

# ── Request / Response models ─────────────────────────────────────────────────
class GuessEntry(BaseModel):
    word: str
    pattern: list[int]

class SuggestRequest(BaseModel):
    history: list[GuessEntry] = []

class SuggestResponse(BaseModel):
    suggestion: str
    top_suggestions: list[dict]
    possible_count: int
    bits_remaining: float
    solved: bool
    message: str

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "opener": OPENING}

@app.post("/suggest", response_model=SuggestResponse)
def suggest(req: SuggestRequest):
    possible = list(ANSWERS)

    for entry in req.history:
        word    = entry.word.lower().strip()
        pattern = tuple(entry.pattern)
        if len(word) != 5:
            raise HTTPException(400, f"Word must be 5 letters: {word}")
        if len(pattern) != 5 or not all(p in (0, 1, 2) for p in pattern):
            raise HTTPException(400, "Pattern must be 5 values of 0, 1, or 2")
        if pattern == WIN_PATTERN:
            return SuggestResponse(
                suggestion=word, top_suggestions=[], possible_count=1,
                bits_remaining=0.0, solved=True,
                message=f"Solved in {len(req.history)} guesses!"
            )
        possible = filter_words(possible, word, pattern)

    if not possible:
        raise HTTPException(422, "No possible words remaining. Check your pattern input.")

    history_tuples = [(e.word.lower(), tuple(e.pattern)) for e in req.history]
    suggestion     = model_suggest(history_tuples, possible)
    top_suggs      = top_suggestions(history_tuples, possible)
    bits           = math.log2(len(possible)) if len(possible) > 1 else 0.0

    return SuggestResponse(
        suggestion=suggestion,
        top_suggestions=top_suggs,
        possible_count=len(possible),
        bits_remaining=round(bits, 2),
        solved=False,
        message=f"{len(possible)} words remaining — try {suggestion.upper()}"
    )

@app.get("/opener")
def get_opener():
    return {"word": OPENING}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)