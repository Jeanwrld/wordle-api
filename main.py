app_code = '''import json, math, torch, gradio as gr
from collections import Counter
import numpy as np
from huggingface_hub import hf_hub_download
import torch.nn as nn

REPO_ID = "sato2ru/wordle-solver"
HF_REPO_ID = "sato2ru/wordle-solver"
config  = json.load(open(hf_hub_download(REPO_ID, "config.json")))
ANSWERS = json.load(open(hf_hub_download(REPO_ID, "answers.json")))
ALLOWED = json.load(open(hf_hub_download(REPO_ID, "allowed.json")))
WORD2IDX = {w: i for i, w in enumerate(ALLOWED)}
LETTERS  = "abcdefghijklmnopqrstuvwxyz"
L2I = {c: i for i, c in enumerate(LETTERS)}
INPUT_DIM  = config["input_dim"]
OUTPUT_DIM = config["output_dim"]
OPENING    = config["opening_guess"]
WIN_PATTERN = (2,2,2,2,2)

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
model.load_state_dict(torch.load(hf_hub_download(REPO_ID, "model_weights.pt"), map_location="cpu"))
model.eval()

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
    return [w for w in words if get_pattern(guess, w) == pattern]

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
    if len(possible) == 1:
        return possible[0]
    if not history:
        return OPENING
    state = torch.tensor(encode_board(history)).unsqueeze(0)
    with torch.no_grad():
        logits = model(state)[0]
    top5 = [ALLOWED[i] for i in logits.topk(5).indices.tolist()]
    return max(top5, key=lambda w: entropy_score(w, possible))

def render_board(history):
    colours = {0: "⬜", 1: "🟨", 2: "🟩"}
    rows = []
    for word, pattern in history:
        tiles = "  ".join(colours[s] + c.upper() for c, s in zip(word, pattern))
        rows.append(tiles)
    return "\\n".join(rows) if rows else "(no guesses yet)"

def process_guess(guess_input, pattern_input, state):
    if state["done"]:
        return render_board(state["history"]), "Game over — press Reset", state

    guess = guess_input.strip().lower()
    if len(guess) != 5:
        return render_board(state["history"]), "Word must be 5 letters", state
    if len(pattern_input) != 5 or not all(c in "012" for c in pattern_input):
        return render_board(state["history"]), "Pattern must be 5 digits (0/1/2)", state

    pattern = tuple(int(c) for c in pattern_input)
    state["history"].append((guess, pattern))

    if pattern == WIN_PATTERN:
        state["done"] = True
        turns = len(state["history"])
        return render_board(state["history"]), f"Solved in {turns} turns!", state

    state["possible"] = filter_words(state["possible"], guess, pattern)

    if not state["possible"]:
        state["done"] = True
        return render_board(state["history"]), "No words left. Check your input.", state

    suggestion = model_suggest(state["history"], state["possible"])
    remaining = len(state["possible"])
    msg = f"Try: {suggestion.upper()}  |  {remaining} words remaining"
    return render_board(state["history"]), msg, state

def reset_game(state):
    new_state = {"possible": list(ANSWERS), "history": [], "done": False}
    return render_board([]), f"Try: {OPENING.upper()} to start", new_state

def init_state():
    return {"possible": list(ANSWERS), "history": [], "done": False}

with gr.Blocks(title="Wordle Solver", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# Wordle Solver")
    gr.Markdown("Enter each guess and the colour pattern. **0** = grey  **1** = yellow  **2** = green")

    state    = gr.State(init_state())
    board    = gr.Textbox(label="Board", lines=8, interactive=False)
    message  = gr.Markdown(f"Try: **{OPENING.upper()}** to start")

    with gr.Row():
        guess_box   = gr.Textbox(label="Your guess", placeholder="crane", max_lines=1)
        pattern_box = gr.Textbox(label="Pattern", placeholder="00000", max_lines=1)

    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
        reset_btn  = gr.Button("Reset")

    submit_btn.click(process_guess, [guess_box, pattern_box, state], [board, message, state])
    reset_btn.click(reset_game, [state], [board, message, state])

demo.launch()
'''

with open('app.py', 'w') as f:
    f.write(app_code)

with open('requirements.txt', 'w') as f:
    f.write('torch\ngradio\nhuggingface_hub\nnumpy\n')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
