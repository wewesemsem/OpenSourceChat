# Local Chat

A chat page for models running on your own computer. Conversations stay in a local database. The app talks to [Ollama](https://ollama.com) on this machine and does not send chats to a hosted model.

The shipped setup offers **Gemma 3 12B**, **Llama 3.1 8B**, and **Gemma 3 4B (fast)**. The larger models use an 8,192-token context. The fast model uses 4,096, which is easier on a machine without a GPU.

## What you need

- Python 3.11 or newer
- Ollama, running locally
- Enough free memory for the model you select. Gemma 3 12B is the heaviest. Gemma 3 4B is the one to pick when replies feel slow.

## 1. Install Ollama and the models

Install Ollama from [ollama.com](https://ollama.com), start it, then pull the models named in `config.yaml`:

```bash
ollama pull gemma3:12b
ollama pull llama3.1:8b
ollama pull gemma3:4b
```

Check the names with `ollama list`. Each `name` under `models` in `config.yaml` must match that list exactly, including the tag (`gemma3:12b`, not `gemma3`).

Ollama listens on `http://127.0.0.1:11434` by default. If you moved it, change `base_url` for each model. The app refuses a model address that is not on this computer unless you set `allow_remote_models: true`.

To use only one model, delete the other entry and set `default_model` to the one you kept.

## 2. Install the app

```bash
git clone https://github.com/wewesemsem/OpenSourceChat.git
cd OpenSourceChat
python3 -m venv .venv
```

Activate the virtual environment:

```bash
# macOS and Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Then install it:

```bash
pip install -e .
```

## 3. Start it

```bash
python -m app
```

`local-chat` is the same command after the install above.

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The model menu lists whatever is in `config.yaml`. A model shows as ready when Ollama is running and that model is pulled. Choosing a model loads it, and `keep_alive` in `config.yaml` keeps it loaded for 30 minutes so the next reply skips that wait.

Each reply shows two timings: how long the first word took, and how fast the rest arrived. A long wait before the first word is model load or a long prompt. A low tok/s after that is generation speed.

Chats are stored in `data/chat.sqlite`. That folder is not part of the git repo, so a new computer starts with an empty history.

## Attach a file

**Attach file** opens the system file window. The page does not receive the folder path. The app keeps a copy under `data/uploads` and reads that copy on later turns. **Remove file** deletes the copy, not the original.

Allowed types are `.txt`, `.md`, `.pdf`, `.docx`, and `.csv`, up to 20 MB. Hidden files, `.env`, and key files are refused.

## Index a folder

**Select Folder** opens the system folder window. You can choose any folder this user can read. `files.allowed_folders` does not limit that choice, and `index.enabled` does not have to be turned on. The same type, size, hidden-file, and link rules apply, and a folder that is too large is left unindexed.

The command `python -m app.index index <folder>` still requires `index.enabled` and a folder listed in `files.allowed_folders`.

## If you change the address

The server binds to `127.0.0.1` port `8000`. To use another port, edit `server.port` in `config.yaml` and update `security.allowed_origins` to the same port, for example `http://127.0.0.1:9000` and `http://localhost:9000`.

Leave `server.host` as `127.0.0.1` when you are the only person using it. Opening it on a shared network means setting `auth_token` in `config.yaml` and listing the hostnames you intend to allow.

## Tests

```bash
python -m unittest discover -s tests -q
```
