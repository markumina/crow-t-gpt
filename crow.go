// crow.go — the entire backend for Crow T. Robot, in one file.
//
// Build:  go build -o crow crow.go
// Run:    ./crow          then open http://localhost:8765
//
// Uses ONLY the Go standard library. The heavy lifting is done by
// external programs that never rot:
//   - whisper.cpp  (speech to text, local binary)
//   - piper + sox  (text to speech + pitch shift, local binaries)
//   - ollama       (the brain, local HTTP server)
package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"sync"
	"time"
)

// ----------------------------------------------------------------------------
// CONFIG — edit paths here if you put things somewhere else
// ----------------------------------------------------------------------------

var (
	port         = "8765"
	whisperBin   = firstExisting("./whisper.cpp/build/bin/whisper-cli", "./whisper.cpp/build/bin/main", "./whisper.cpp/main")
	whisperModel = "./whisper.cpp/models/ggml-base.en.bin"
	piperBin     = "./piper/piper"
	piperVoice   = "./voices/en_US-ryan-medium.onnx"
	ollamaURL    = "http://localhost:11434"
	ollamaModel  = "qwen2.5:7b"

	// Crow's voice character: sox pitch is in cents (100 = one semitone).
	// Raise pitch for the nasal quality, nudge tempo up because Crow talks fast.
	soxPitch = "380"
	soxTempo = "1.05"

	whisperThreads = "8"
	maxToolRounds  = 4
	maxHistory     = 40 // messages kept after the system prompt
)

// ----------------------------------------------------------------------------
// Personality
// ----------------------------------------------------------------------------

const defaultPersonality = `You are Crow T. Robot, a sarcastic gold robot puppet. Keep replies short, spoken-style, and funny. No emoji, no markdown, no stage directions. Use web_search for anything current, then answer in character.`

func loadPersonality() string {
	b, err := os.ReadFile("personality.txt")
	if err != nil || len(strings.TrimSpace(string(b))) == 0 {
		return defaultPersonality
	}
	return string(b)
}

// ----------------------------------------------------------------------------
// Ollama chat types
// ----------------------------------------------------------------------------

type Msg struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

type ToolCall struct {
	Function FuncCall `json:"function"`
}

type FuncCall struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

type chatReq struct {
	Model    string                 `json:"model"`
	Messages []Msg                  `json:"messages"`
	Tools    json.RawMessage        `json:"tools,omitempty"`
	Stream   bool                   `json:"stream"`
	Options  map[string]interface{} `json:"options,omitempty"`
}

type chatResp struct {
	Message Msg    `json:"message"`
	Done    bool   `json:"done"`
	Error   string `json:"error"`
}

const toolsJSON = `[
  {"type":"function","function":{
    "name":"web_search",
    "description":"Search the web for current information: news, weather, prices, sports, recent events, anything after your training data or that you are not sure about.",
    "parameters":{"type":"object","properties":{"query":{"type":"string","description":"the search query"}},"required":["query"]}}},
  {"type":"function","function":{
    "name":"fetch_page",
    "description":"Fetch a web page by URL and return its readable text. Use after web_search when a result looks promising and you need details.",
    "parameters":{"type":"object","properties":{"url":{"type":"string","description":"full http or https URL"}},"required":["url"]}}}
]`

var (
	ollamaClient = &http.Client{Timeout: 300 * time.Second}
	webClient    = &http.Client{Timeout: 20 * time.Second}
)

func ollamaChat(msgs []Msg, withTools bool) (Msg, error) {
	req := chatReq{
		Model:    ollamaModel,
		Messages: msgs,
		Stream:   false,
		Options:  map[string]interface{}{"num_ctx": 8192, "temperature": 0.85},
	}
	if withTools {
		req.Tools = json.RawMessage(toolsJSON)
	}
	body, err := json.Marshal(req)
	if err != nil {
		return Msg{}, err
	}
	resp, err := ollamaClient.Post(ollamaURL+"/api/chat", "application/json", bytes.NewReader(body))
	if err != nil {
		return Msg{}, fmt.Errorf("can't reach ollama at %s — is it running? (%v)", ollamaURL, err)
	}
	defer resp.Body.Close()
	var out chatResp
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return Msg{}, fmt.Errorf("bad response from ollama: %v", err)
	}
	if out.Error != "" {
		return Msg{}, fmt.Errorf("ollama error: %s", out.Error)
	}
	return out.Message, nil
}

// ----------------------------------------------------------------------------
// Tools: web search (DuckDuckGo HTML, no API key) + page fetch
// ----------------------------------------------------------------------------

var (
	reResultLink = regexp.MustCompile(`(?s)<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>`)
	reSnippet    = regexp.MustCompile(`(?s)class="result__snippet"[^>]*>(.*?)</a>`)
	reScript     = regexp.MustCompile(`(?is)<script.*?</script>`)
	reStyle      = regexp.MustCompile(`(?is)<style.*?</style>`)
	reTag        = regexp.MustCompile(`(?s)<[^>]*>`)
	reSpace      = regexp.MustCompile(`\s+`)
	reBrackets   = regexp.MustCompile(`\[[^\]]*\]|\([^)]*\)`)
)

var htmlUnescaper = strings.NewReplacer(
	"&amp;", "&", "&lt;", "<", "&gt;", ">", "&quot;", `"`,
	"&#x27;", "'", "&#39;", "'", "&nbsp;", " ",
)

func stripHTML(s string) string {
	s = reScript.ReplaceAllString(s, " ")
	s = reStyle.ReplaceAllString(s, " ")
	s = reTag.ReplaceAllString(s, " ")
	s = htmlUnescaper.Replace(s)
	return strings.TrimSpace(reSpace.ReplaceAllString(s, " "))
}

// DuckDuckGo wraps result links as /l/?uddg=<encoded-real-url>&rut=...
func realURL(href string) string {
	u, err := url.Parse(href)
	if err != nil {
		return href
	}
	if dest := u.Query().Get("uddg"); dest != "" {
		return dest
	}
	return href
}

func webSearch(query string) string {
	req, err := http.NewRequest("GET", "https://html.duckduckgo.com/html/?q="+url.QueryEscape(query), nil)
	if err != nil {
		return "search failed: " + err.Error()
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) CrowT2000/2.0")
	resp, err := webClient.Do(req)
	if err != nil {
		return "search failed: " + err.Error()
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Sprintf("search failed: DuckDuckGo returned HTTP %d", resp.StatusCode)
	}
	page, err := io.ReadAll(io.LimitReader(resp.Body, 800_000))
	if err != nil {
		return "search failed: " + err.Error()
	}
	links := reResultLink.FindAllStringSubmatch(string(page), 6)
	snips := reSnippet.FindAllStringSubmatch(string(page), 6)
	if len(links) == 0 {
		return "search returned no results for: " + query
	}
	var b strings.Builder
	for i, m := range links {
		if i >= 5 {
			break
		}
		fmt.Fprintf(&b, "%d. %s\n   URL: %s\n", i+1, stripHTML(m[2]), realURL(m[1]))
		if i < len(snips) {
			fmt.Fprintf(&b, "   %s\n", stripHTML(snips[i][1]))
		}
	}
	return b.String()
}

func fetchPage(raw string) string {
	if !strings.HasPrefix(raw, "http://") && !strings.HasPrefix(raw, "https://") {
		return "fetch failed: URL must start with http:// or https://"
	}
	req, err := http.NewRequest("GET", raw, nil)
	if err != nil {
		return "fetch failed: " + err.Error()
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) CrowT2000/2.0")
	resp, err := webClient.Do(req)
	if err != nil {
		return "fetch failed: " + err.Error()
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Sprintf("fetch failed: HTTP %d", resp.StatusCode)
	}
	page, err := io.ReadAll(io.LimitReader(resp.Body, 500_000))
	if err != nil {
		return "fetch failed: " + err.Error()
	}
	text := stripHTML(string(page))
	runes := []rune(text)
	if len(runes) > 4000 {
		text = string(runes[:4000]) + " …[truncated]"
	}
	if text == "" {
		return "fetch succeeded but the page had no readable text"
	}
	return text
}

func argString(m map[string]interface{}, key string) string {
	if v, ok := m[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
		return fmt.Sprint(v)
	}
	return ""
}

func runTool(tc ToolCall) string {
	switch tc.Function.Name {
	case "web_search":
		return webSearch(argString(tc.Function.Arguments, "query"))
	case "fetch_page":
		return fetchPage(argString(tc.Function.Arguments, "url"))
	default:
		return "unknown tool: " + tc.Function.Name
	}
}

// ----------------------------------------------------------------------------
// Conversation state + the brain loop
// ----------------------------------------------------------------------------

var (
	turnMu  sync.Mutex // one turn at a time — it's a one-robot show
	history []Msg
)

func resetHistory() {
	history = []Msg{{Role: "system", Content: loadPersonality()}}
}

func trimHistory() {
	if len(history) > maxHistory+1 {
		trimmed := []Msg{history[0]}
		history = append(trimmed, history[len(history)-maxHistory:]...)
	}
}

// brainReply runs the tool-calling loop and returns Crow's final line.
func brainReply(userText string) (string, []string, error) {
	history = append(history, Msg{Role: "user", Content: userText})
	trimHistory()

	msgs := make([]Msg, len(history))
	copy(msgs, history)

	var toolsUsed []string
	for round := 0; round < maxToolRounds; round++ {
		reply, err := ollamaChat(msgs, true)
		if err != nil {
			return "", toolsUsed, err
		}
		if len(reply.ToolCalls) == 0 {
			final := strings.TrimSpace(reply.Content)
			history = append(history, Msg{Role: "assistant", Content: final})
			return final, toolsUsed, nil
		}
		msgs = append(msgs, reply)
		for _, tc := range reply.ToolCalls {
			toolsUsed = append(toolsUsed, tc.Function.Name)
			fmt.Printf("  [tool] %s(%v)\n", tc.Function.Name, tc.Function.Arguments)
			result := runTool(tc)
			msgs = append(msgs, Msg{Role: "tool", Content: fmt.Sprintf("Result of %s:\n%s", tc.Function.Name, result)})
		}
	}
	// Ran out of tool rounds — force a plain answer with what we have.
	reply, err := ollamaChat(msgs, false)
	if err != nil {
		return "", toolsUsed, err
	}
	final := strings.TrimSpace(reply.Content)
	history = append(history, Msg{Role: "assistant", Content: final})
	return final, toolsUsed, nil
}

// ----------------------------------------------------------------------------
// Speech to text (whisper.cpp) and text to speech (piper + sox)
// ----------------------------------------------------------------------------

func transcribe(wav []byte) (string, error) {
	tmp, err := os.CreateTemp("", "crow-in-*.wav")
	if err != nil {
		return "", err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(wav); err != nil {
		tmp.Close()
		return "", err
	}
	tmp.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, whisperBin,
		"-m", whisperModel, "-f", tmp.Name(), "-nt", "-t", whisperThreads)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("whisper failed: %v — %s", err, lastLine(stderr.String()))
	}
	text := strings.TrimSpace(stdout.String())
	// Whisper labels non-speech like [BLANK_AUDIO] or (wind blowing); strip it.
	text = strings.TrimSpace(reBrackets.ReplaceAllString(text, ""))
	text = strings.TrimSpace(reSpace.ReplaceAllString(text, " "))
	return text, nil
}

func speak(text string) ([]byte, error) {
	raw, err := os.CreateTemp("", "crow-raw-*.wav")
	if err != nil {
		return nil, err
	}
	raw.Close()
	defer os.Remove(raw.Name())

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, piperBin, "--model", piperVoice, "--output_file", raw.Name())
	cmd.Stdin = strings.NewReader(text)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("piper failed: %v — %s", err, lastLine(stderr.String()))
	}

	outPath := raw.Name()
	if soxOK() {
		shifted, err := os.CreateTemp("", "crow-out-*.wav")
		if err == nil {
			shifted.Close()
			defer os.Remove(shifted.Name())
			sctx, scancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer scancel()
			sox := exec.CommandContext(sctx, "sox", raw.Name(), shifted.Name(), "pitch", soxPitch, "tempo", soxTempo)
			if err := sox.Run(); err == nil {
				outPath = shifted.Name()
			} else {
				fmt.Println("  [warn] sox pitch shift failed, using unshifted voice")
			}
		}
	}
	return os.ReadFile(outPath)
}

func soxOK() bool {
	_, err := exec.LookPath("sox")
	return err == nil
}

func lastLine(s string) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	if len(lines) == 0 {
		return ""
	}
	return lines[len(lines)-1]
}

// ----------------------------------------------------------------------------
// HTTP handlers
// ----------------------------------------------------------------------------

func writeJSON(w http.ResponseWriter, code int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func handleTranscribe(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, 405, map[string]string{"error": "POST only"})
		return
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 32<<20))
	if err != nil {
		writeJSON(w, 400, map[string]string{"error": "couldn't read audio: " + err.Error()})
		return
	}
	text, err := transcribe(body)
	if err != nil {
		fmt.Println("  [error]", err)
		writeJSON(w, 500, map[string]string{"error": err.Error()})
		return
	}
	if text != "" {
		fmt.Printf("[heard] %s\n", text)
	}
	writeJSON(w, 200, map[string]string{"text": text})
}

func handleReply(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, 405, map[string]string{"error": "POST only"})
		return
	}
	var in struct {
		Text string `json:"text"`
	}
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil || strings.TrimSpace(in.Text) == "" {
		writeJSON(w, 400, map[string]string{"error": "need JSON like {\"text\":\"...\"}"})
		return
	}

	turnMu.Lock()
	defer turnMu.Unlock()

	reply, toolsUsed, err := brainReply(strings.TrimSpace(in.Text))
	if err != nil {
		fmt.Println("  [error]", err)
		writeJSON(w, 500, map[string]string{"error": err.Error()})
		return
	}
	fmt.Printf("[crow] %s\n", reply)

	audio, err := speak(reply)
	if err != nil {
		fmt.Println("  [error]", err)
		// Still return the text so the UI can show it even if the voice broke.
		writeJSON(w, 200, map[string]interface{}{
			"reply": reply, "audio": "", "tools": toolsUsed,
			"error": "voice failed: " + err.Error(),
		})
		return
	}
	writeJSON(w, 200, map[string]interface{}{
		"reply": reply,
		"audio": base64.StdEncoding.EncodeToString(audio),
		"tools": toolsUsed,
	})
}

func handleReset(w http.ResponseWriter, r *http.Request) {
	turnMu.Lock()
	defer turnMu.Unlock()
	resetHistory()
	fmt.Println("[reset] conversation cleared")
	writeJSON(w, 200, map[string]string{"ok": "true"})
}

func handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	http.ServeFile(w, r, "index.html")
}

// ----------------------------------------------------------------------------
// Boot diagnostics — the launcher script, built in
// ----------------------------------------------------------------------------

func firstExisting(paths ...string) string {
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return paths[0]
}

func check(name string, ok bool, fix string) bool {
	mark := "✓"
	if !ok {
		mark = "✗"
	}
	fmt.Printf("  %s %s\n", mark, name)
	if !ok {
		fmt.Printf("      fix: %s\n", fix)
	}
	return ok
}

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

func ollamaHasModel() (running bool, hasModel bool) {
	resp, err := (&http.Client{Timeout: 3 * time.Second}).Get(ollamaURL + "/api/tags")
	if err != nil {
		return false, false
	}
	defer resp.Body.Close()
	var tags struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&tags); err != nil {
		return true, false
	}
	for _, m := range tags.Models {
		if strings.HasPrefix(m.Name, ollamaModel) {
			return true, true
		}
	}
	return true, false
}

func main() {
	fmt.Println()
	fmt.Println("  ┌─────────────────────────────────────────┐")
	fmt.Println("  │  CROW T. ROBOT 2.0 — pre-flight checks  │")
	fmt.Println("  └─────────────────────────────────────────┘")

	allGood := true
	allGood = check("whisper.cpp binary ("+whisperBin+")", fileExists(whisperBin),
		"git clone https://github.com/ggml-org/whisper.cpp && cd whisper.cpp && cmake -B build && cmake --build build -j") && allGood
	allGood = check("whisper model ("+whisperModel+")", fileExists(whisperModel),
		"cd whisper.cpp && sh ./models/download-ggml-model.sh base.en") && allGood
	allGood = check("piper binary ("+piperBin+")", fileExists(piperBin),
		"download the linux x86_64 tarball from github.com/rhasspy/piper/releases and untar it into ./piper/") && allGood
	allGood = check("piper voice ("+piperVoice+")", fileExists(piperVoice) && fileExists(piperVoice+".json"),
		"put en_US-ryan-medium.onnx AND en_US-ryan-medium.onnx.json in ./voices/") && allGood
	allGood = check("sox (voice pitch shift)", soxOK(),
		"sudo apt install sox   (Crow will sound less Crow-like without it)") && allGood

	running, hasModel := ollamaHasModel()
	allGood = check("ollama running at "+ollamaURL, running, "start it: ollama serve   (or: systemctl start ollama)") && allGood
	if running {
		allGood = check("model "+ollamaModel+" pulled", hasModel, "ollama pull "+ollamaModel) && allGood
	}
	check("index.html present", fileExists("index.html"), "keep index.html in the same folder as this binary") // non-fatal-ish
	check("personality.txt present", fileExists("personality.txt"), "optional — using built-in fallback personality")

	fmt.Println()
	if allGood {
		fmt.Println("  All systems nominal. Which, on this satellite, is suspicious.")
	} else {
		fmt.Println("  Some checks failed — the server will still start, but fix the ✗ items above.")
	}
	fmt.Printf("\n  → open http://localhost:%s and hit POWER\n\n", port)

	resetHistory()

	http.HandleFunc("/", handleIndex)
	http.HandleFunc("/api/transcribe", handleTranscribe)
	http.HandleFunc("/api/reply", handleReply)
	http.HandleFunc("/api/reset", handleReset)

	if err := http.ListenAndServe("127.0.0.1:"+port, nil); err != nil {
		fmt.Println("server failed:", err)
		os.Exit(1)
	}
}
