# CROW T. ROBOT 2.0

One Go file, one HTML file for the face, and three local engines:
whisper.cpp (ears), piper + sox (voice), ollama (brain).

Folder layout:

```
crow/
├── crow.go            ← the whole backend
├── index.html         ← the whole frontend
├── personality.txt    ← Crow's brain flavor (edit freely, restart to apply)
├── whisper.cpp/       ← built from source, once
├── piper/             ← prebuilt binary, untarred
└── voices/            ← en_US-ryan-medium.onnx + .onnx.json
```

## 1. System packages (one line, apt only)

```bash
sudo apt update && sudo apt install -y golang-go sox git build-essential cmake wget
```

## 2. Ears — whisper.cpp (build once, ~2 min)

```bash
cd ~/crow
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp
cmake -B build && cmake --build build -j
sh ./models/download-ggml-model.sh base.en
cd ..
```

## 3. Voice — piper binary + the Ryan voice

```bash
mkdir -p piper voices
wget -O /tmp/piper.tar.gz https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz
tar -xzf /tmp/piper.tar.gz -C .
```

(If that URL has moved, grab the linux x86_64 tarball from
github.com/rhasspy/piper/releases — any recent one works.)

Voice files: **if you still have `en_US-ryan-medium.onnx` and
`en_US-ryan-medium.onnx.json` from the Python version, just copy them into
`./voices/` and skip the downloads.** Otherwise:

```bash
wget -P voices https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx
wget -P voices https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json
```

Once these are on disk they're vendored - no network, for now.

## 4. Brain — ollama

```bash
ollama pull qwen2.5:7b   # skip if already pulled
```

## 5. Build and run

```bash
go build -o crow crow.go
./crow
```

It runs its own pre-flight check and tells you what's missing and the
command to fix it. When it's happy, open **http://localhost:8765**, hit
**POWER**, allow the mic, and talk.

## Tuning

- **Voice character**: `soxPitch` and `soxTempo` at the top of `crow.go`
  (380 cents up, 1.05× speed by default). Rebuild after editing.
- **Personality**: edit `personality.txt`, restart `./crow`. No rebuild needed.
- **Mic pickiness**: the MIC SENS slider in the UI, live.
- **Face**: everything visual is in `index.html` — edit and refresh the
  browser. No rebuild needed.

## Notes

- whisper.cpp, piper, and the models are plain files sitting in this folder.
- The frontend is one dependency-free HTML file running in browser.
  exists.
- The only moving part is ollama

