from fastapi import FastAPI, HTTPException
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

print("Loading model...")
config  = json.load(open(hf_hub_download(HF_REPO_ID, "config.json")))
ANSWERS = json.load(open(hf_hub_download(HF_REPO_ID, "answers.json")))
ALLOWED = json.load(open(hf_hub_download(HF_REPO_ID, "allowed.json")))
WORD2IDX = {w: i for i, w in enumerate(ALLOWED)}
LETTERS  = "abcdefghijklmnopqrstuvwxyz"
L2I = {c: i for i, c in enumerate(LETTERS)}
INPUT_DIM  = config["input_dim"]
OUTPUT_DIM = config["output_dim"]
OPENING    = config["opening_guess"]
WIN_PATTERN = (2, 2, 2, 2, 2)

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
print("Model loaded")

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

def model_suggest(history, possible):
    if not possible:       return None
    if len(possible) == 1: return possible[0]
    if not history:        return OPENING

    # ── Late-game elimination mode ────────────────────────────────
    # When few words remain, find a guess that eliminates the most
    # candidates by targeting ambiguous letters at ambiguous positions
    if len(possible) <= 6:
        # Find positions where letters still vary
        ambiguous_letters = set()
        for pos in range(5):
            letters_at_pos = set(w[pos] for w in possible)
            if len(letters_at_pos) > 1:
                ambiguous_letters.update(letters_at_pos)

        # Find a guess (from all allowed words) that covers the most
        # ambiguous letters in one go
        best_elim, best_score = None, -1
        for g in ALLOWED:
            if g in possible and len(possible) > 2:
                continue  # don't guess a possible answer unless only 1-2 left
            score = len(set(g) & ambiguous_letters)
            h = entropy_score(g, possible)
            combined = score * 2 + h  # weight elimination + entropy
            if combined > best_score:
                best_score, best_elim = combined, g

        if best_elim:
            return best_elim

    # ── Normal model inference ────────────────────────────────────
    state = torch.tensor(encode_board(history)).unsqueeze(0)
    with torch.no_grad():
        logits = model(state)[0]
    top5 = [ALLOWED[i] for i in logits.topk(5).indices.tolist()]
    return max(top5, key=lambda w: entropy_score(w, possible))

def top_suggestions(history, possible, n=5):
    if not possible: return []
    if not history:
        candidates = [OPENING] + [w for w in ALLOWED if w != OPENING][:20]
    else:
        state = torch.tensor(encode_board(history)).unsqueeze(0)
        with torch.no_grad():
            logits = model(state)[0]
        candidates = [ALLOWED[i] for i in logits.topk(20).indices.tolist()]
    possible_set = set(possible)
    scored = [{"word": w, "entropy": round(entropy_score(w, possible), 3), "is_possible": w in possible_set} for w in candidates]
    scored.sort(key=lambda x: (-x["entropy"], not x["is_possible"]))
    return scored[:n]

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
        if len(pattern) != 5 or not all(p in (0,1,2) for p in pattern):
            raise HTTPException(400, "Pattern must be 5 values of 0, 1, or 2")
        if pattern == WIN_PATTERN:
            return SuggestResponse(
                suggestion=word, top_suggestions=[], possible_count=1,
                bits_remaining=0.0, solved=True,
                message=f"Solved in {len(req.history)} guesses!"
            )
        possible = filter_words(possible, word, pattern)

    if not possible:
        raise HTTPException(422, "No possible words remaining.")

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
