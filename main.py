from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json, math, torch, numpy as np
from collections import Counter
from typing import Optional
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

# ── Load word lists & configs ─────────────────────────────────────────────────
print("Loading configs and word lists...")
config    = json.load(open(hf_hub_download(HF_REPO_ID, "config.json")))
rl_config = json.load(open(hf_hub_download(HF_REPO_ID, "rl_config.json")))
ANSWERS  = json.load(open(hf_hub_download(HF_REPO_ID, "answers.json")))
ALLOWED  = json.load(open(hf_hub_download(HF_REPO_ID, "allowed.json")))
WORD2IDX = {w: i for i, w in enumerate(ALLOWED)}
LETTERS  = "abcdefghijklmnopqrstuvwxyz"
L2I      = {c: i for i, c in enumerate(LETTERS)}
INPUT_DIM  = config["input_dim"]
OUTPUT_DIM = config["output_dim"]
OPENING    = config["opening_guess"]
WIN_PATTERN = (2, 2, 2, 2, 2)


# ── Model architecture ────────────────────────────────────────────────────────
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


class RLWordleNet(nn.Module):
    """Same encoder as WordleNet but with BatchNorm-safe single-sample forward."""
    def __init__(self):
        super().__init__()
        h = rl_config["hidden"]
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h, h),         nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h, 256),       nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, OUTPUT_DIM)

    def forward(self, x):
        single = x.shape[0] == 1
        if single:
            x = x.repeat(2, 1)
        feat   = self.encoder(x)
        if single:
            feat = feat[:1]
        return self.policy_head(feat)


# ── Load weights ──────────────────────────────────────────────────────────────
print("Loading supervised model...")
model = WordleNet()
model.load_state_dict(
    torch.load(hf_hub_download(HF_REPO_ID, "model_weights.pt"), map_location="cpu")
)
model.eval()

print("Loading RL model...")
rl_model = RLWordleNet()
rl_model.load_state_dict(
    torch.load(hf_hub_download(HF_REPO_ID, "rl_model_weights.pt"), map_location="cpu"),strict=False
)
rl_model.eval()
print("Both models loaded.")


# ── Core logic ────────────────────────────────────────────────────────────────
def get_pattern(guess, answer):
    pattern = [0]*5
    counts = Counter(answer)
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
    return sum(-(c/n)*math.log2(c/n) for c in buckets.values())

def encode_board(history):
    vec = np.zeros(INPUT_DIM, dtype=np.float32)
    for word, pattern in history:
        for pos, (letter, state) in enumerate(zip(word, pattern)):
            vec[L2I[letter]*15 + pos*3 + state] = 1.0
    return vec

def get_logits(history, possible, use_rl=False):
    """Get top-20 model words using constraint-filtered mask."""
    active_model = rl_model if use_rl else model
    possible_set = set(possible)

    state = torch.tensor(encode_board(history)).unsqueeze(0)
    with torch.no_grad():
        logits = active_model(state)[0]

    # For RL model: mask to constraint-consistent words before taking top-20
    if use_rl:
        mask = torch.full((OUTPUT_DIM,), float('-inf'))
        for i, w in enumerate(ALLOWED):
            if w in possible_set:
                mask[i] = 0.0
        if mask.max() == float('-inf'):
            mask[:] = 0.0  # fallback: no valid words found, unmask all
        logits = logits + mask

    return [ALLOWED[i] for i in logits.topk(20).indices.tolist()]


def model_suggest(history, possible, use_rl=False):
    if not possible:       return None
    if len(possible) == 1: return possible[0]
    if not history:        return OPENING
    guessed = {w for w, _ in history}

    model_words = get_logits(history, possible, use_rl)

    if len(possible) <= 6:
        best, best_worst = None, float('inf')
        for g in list(possible) + model_words:
            if g in guessed: continue
            worst = max(Counter(get_pattern(g, w) for w in possible).values())
            if worst < best_worst:
                best_worst, best = worst, g
        return best

    candidates = list(dict.fromkeys(model_words + list(possible)))
    candidates = [w for w in candidates if w not in guessed]
    if not candidates: return possible[0]
    return max(candidates, key=lambda w: entropy_score(w, possible))


def top_suggestions(history, possible, use_rl=False, n=5):
    if not possible: return []
    guessed = {w for w, _ in history}
    if not history:
        candidates = [OPENING] + [w for w in ALLOWED if w != OPENING][:20]
    else:
        model_words = get_logits(history, possible, use_rl)
        candidates  = list(dict.fromkeys(model_words + list(possible)))

    possible_set = set(possible)
    candidates = [w for w in candidates if w in possible_set and w not in guessed]
    scored = [{"word": w, "entropy": round(entropy_score(w, possible), 3), "is_possible": True}
              for w in candidates]
    scored.sort(key=lambda x: -x["entropy"])
    return scored[:n]


# ── Schemas ───────────────────────────────────────────────────────────────────
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
    model_used: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "opener": OPENING}

@app.post("/suggest", response_model=SuggestResponse)
def suggest(
    req: SuggestRequest,
    model: str = Query(default="supervised", pattern="^(supervised|rl)$")
):
    use_rl   = model == "rl"
    possible = list(ANSWERS)

    for entry in req.history:
        word    = entry.word.lower().strip()
        pattern = tuple(entry.pattern)
        if len(word) != 5:
            raise HTTPException(400, f"Word must be 5 letters: {word}")
        if len(pattern) != 5 or not all(p in (0,1,2) for p in pattern):
            raise HTTPException(400, "Pattern must be 5 values of 0, 1, or 2")
        if pattern == WIN_PATTERN:
            return SuggestResponse(
                suggestion=word, top_suggestions=[], possible_count=1,
                bits_remaining=0.0, solved=True, model_used=model,
                message=f"Solved in {len(req.history)} guesses!"
            )
        possible = filter_words(possible, word, pattern)

    if not possible:
        raise HTTPException(422, "No possible words remaining.")

    history_tuples = [(e.word.lower(), tuple(e.pattern)) for e in req.history]
    suggestion     = model_suggest(history_tuples, possible, use_rl=use_rl)
    top_suggs      = top_suggestions(history_tuples, possible, use_rl=use_rl)
    bits           = math.log2(len(possible)) if len(possible) > 1 else 0.0

    return SuggestResponse(
        suggestion=suggestion,
        top_suggestions=top_suggs,
        possible_count=len(possible),
        bits_remaining=round(bits, 2),
        solved=False,
        model_used=model,
        message=f"{len(possible)} words remaining — try {suggestion.upper()}"
    )

@app.get("/opener")
def get_opener():
    return {"word": OPENING}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)