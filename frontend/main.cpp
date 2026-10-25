// frontend/main.cpp - UnbiasedStrata's front end: tokenizer, chat template and an OpenAI-style HTTP endpoint in
// front of the unmodified Strata engine.
//
// One binary, two roles.  Started normally it is the front end: it starts a copy of itself with `--engine`, which
// runs the engine's own program (frontend/engine_entry.cpp) in serve mode, and talks to it over a pipe in the
// engine's line protocol (token ids in, token ids out).  That is the process model upstream's Python server uses,
// so the engine runs exactly as upstream tested it.
//
// The tokenizer and the chat template come from llama.cpp (the pinned submodule): the GGUF's own vocabulary and
// its own Jinja template.  Nothing here opens an outgoing connection.
#include "chat.h"
#include "common.h"
#include "llama.h"

#include <cpp-httplib/httplib.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <list>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

int strata_engine_main(int argc, char** argv);

namespace {

// the id /v1/models and every reply report; --model-name overrides it (requests are not matched against it)
const char* const kDefaultModelName = "qwen3.8-flash-next-iq3_xxs";

struct Options {
    std::string model;                  // shard 1 of the GGUF
    std::string pack;                   // the prepared pack directory
    std::string mtp;                    // the prepared draft-head directory
    std::string host = "127.0.0.1";
    int port = 8080;
    long long ctx = 32768;
    int threads = 0;                    // CPU expert workers; 0 = the engine's default (every physical core)
    std::vector<std::string> extra;     // after `--`: passed to the engine as they are
    // images (optional): the model's own vision encoder (mmproj GGUF) run by the strata-vision helper on the GPU
    std::string vision_mmproj;          // --vision PATH: the mmproj file; empty = text only
    std::string vision_helper;          // --vision-helper PATH: the strata-vision binary (default: beside this one)
    int vision_tokens = 2048;           // --vision-tokens N: the most tokens one picture becomes
    int vision_threads = 0;             // --vision-threads N: the helper's CPU threads (0: its default)
    std::string model_name = kDefaultModelName;   // --model-name NAME
};

void usage() {
    std::fprintf(stderr,
        "usage: unbiased-strata --model SHARD1.gguf --pack DIR --mtp DIR [options] [-- engine options]\n\n"
        "  --model PATH    the model's first GGUF shard (...-00001-of-00002.gguf); the second must be beside it\n"
        "  --pack DIR      the prepared pack for that model\n"
        "  --mtp DIR       the prepared draft head\n"
        "  --ctx N         context length in tokens (default 32768)\n"
        "  --threads N     CPU threads for the experts the GPU does not hold (default: every physical core;\n"
        "                  set it inside a container, where the host's core count is not yours)\n"
        "  --host ADDR     address to listen on (default 127.0.0.1)\n"
        "  --port N        port to listen on (default 8080)\n"
        "  --model-name NAME  the model id /v1/models and replies report (default qwen3.8-flash-next-iq3_xxs)\n"
        "  --vision MMPROJ images: the model's vision encoder (mmproj GGUF), run on the GPU by strata-vision\n"
        "  --vision-tokens N   the most tokens one picture becomes (default 2048; more keeps small text legible)\n"
        "  --vision-helper PATH  the strata-vision binary (default: strata-vision beside this binary)\n"
        "  --vision-threads N  CPU threads for the encoder (default: its own)\n\n"
        "Serves POST /v1/chat/completions (OpenAI style, with streaming), GET /v1/models and GET /health.\n"
        "With --vision, image_url parts take data: URLs and local file paths (JPEG, PNG, GIF, BMP); nothing is\n"
        "fetched from the network.  Leave --vram-reserve-mib room for the encoder in the engine options.\n");
}

bool file_exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

// ---------------------------------------------------------------------------------------------------- the engine

struct GenResult {
    bool ok = false;
    std::string error;
    std::string finish = "length";
    long long produced = 0, prompt_n = 0, draft_accepted = 0, draft_offered = 0;
    double prompt_ms = 0, decode_ms = 0;
};

class Engine {
public:
    bool start(const std::vector<std::string>& args, std::string& err) {
        int to_child[2], from_child[2];
        if (::pipe(to_child) != 0 || ::pipe(from_child) != 0) { err = "pipe failed"; return false; }
        pid_ = ::fork();
        if (pid_ < 0) { err = "fork failed"; return false; }
        if (pid_ == 0) {
            ::dup2(to_child[0], STDIN_FILENO);
            ::dup2(from_child[1], STDOUT_FILENO);
            ::close(to_child[0]); ::close(to_child[1]); ::close(from_child[0]); ::close(from_child[1]);
            std::vector<char*> argv;
            for (const std::string& a : args) argv.push_back(const_cast<char*>(a.c_str()));
            argv.push_back(nullptr);
            ::execv("/proc/self/exe", argv.data());
            std::fprintf(stderr, "unbiased-strata: could not start the engine: %s\n", std::strerror(errno));
            ::_exit(127);
        }
        ::close(to_child[0]);
        ::close(from_child[1]);
        in_fd_ = to_child[1];
        out_ = ::fdopen(from_child[0], "r");
        return out_ != nullptr;
    }

    /// Reads the engine's start-up lines until it says READY.  Loading the model takes minutes.
    bool wait_ready(std::string& err) {
        std::string line;
        while (read_line(line)) {
            if (line.rfind("READY", 0) == 0) return true;
            if (line.rfind("ERR", 0) == 0) { err = line; return false; }
        }
        err = "the engine exited before it was ready (its messages are above)";
        return false;
    }

    /// One request.  `on_token` returns false when the client has gone: the engine is told to stop and the rest of
    /// its output for this request is read and dropped, so the next request starts clean.
    GenResult generate(const std::string& header, const std::vector<llama_token>& ids,
                       const std::function<bool(llama_token)>& on_token) {
        GenResult r;
        std::string req = header;
        req.reserve(header.size() + ids.size() * 7 + 2);
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i) req += ',';
            req += std::to_string(ids[i]);
        }
        req += '\n';
        if (!send(req)) { r.error = "the engine is not running"; return r; }
        bool stopped = false;
        std::string line;
        while (read_line(line)) {
            if (line.size() > 2 && line[0] == 'T' && line[1] == ' ') {
                if (!stopped && !on_token((llama_token) std::atoi(line.c_str() + 2))) {
                    stopped = true;
                    send("STOP\n");
                }
            } else if (line.rfind("DONE ", 0) == 0) {
                char finish[32] = {0};
                std::sscanf(line.c_str(), "DONE %lld %lld %lf %lf %31s %lld %lld", &r.produced, &r.prompt_n, &r.prompt_ms,
                            &r.decode_ms, finish, &r.draft_accepted, &r.draft_offered);
                r.finish = finish;
                r.ok = true;
                return r;
            } else if (line.rfind("ERR", 0) == 0) {
                r.error = line.size() > 4 ? line.substr(4) : "the engine reported an error";
                return r;
            }
            // PP (prompt progress), RESUME, REUSED and INFO lines need no action here
        }
        alive_ = false;
        r.error = "the engine exited during the request";
        return r;
    }

    void quit() {
        if (pid_ > 0) {
            send("QUIT\n");
            int status = 0;
            for (int i = 0; i < 100 && ::waitpid(pid_, &status, WNOHANG) == 0; ++i) ::usleep(100 * 1000);
            ::kill(pid_, SIGKILL);
            ::waitpid(pid_, &status, 0);
            pid_ = -1;
        }
    }

    bool alive() const { return alive_; }
    // one request at a time: the engine holds one conversation.  Recursive: a request with images holds it from the
    // encode on (the encoder and the engine must not use the GPU at the same time, see Vision), then run() takes it
    // again.
    std::recursive_mutex mu;

private:
    bool send(const std::string& s) {
        size_t off = 0;
        while (off < s.size()) {
            const ssize_t n = ::write(in_fd_, s.data() + off, s.size() - off);
            if (n <= 0) { alive_ = false; return false; }
            off += (size_t) n;
        }
        return true;
    }

    bool read_line(std::string& line) {
        char* buf = nullptr;
        size_t cap = 0;
        const ssize_t n = ::getline(&buf, &cap, out_);
        if (n < 0) { std::free(buf); alive_ = false; return false; }
        line.assign(buf, (size_t) n);
        std::free(buf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        return true;
    }

    pid_t pid_ = -1;
    int in_fd_ = -1;
    FILE* out_ = nullptr;
    std::atomic<bool> alive_{true};
};

// ------------------------------------------------------------------------------------------------ the images
//
// A picture reaches the model as rows of embeddings in place of a run of <|image_pad|> tokens.  The rows come from
// the model's own vision encoder (the mmproj GGUF), run by upstream's `strata-vision` helper (tools/vision, built
// on llama.cpp's mtmd) as a resident child process: "ENC <image> <out>" in, "OK <n_tokens> <nx> <ny> <ms>" out, the
// embeddings in a small record file.  The engine, started with --vision, takes "GENI <max_new> <keys> <file> <ids>"
// and places each record's rows at the next run of pad tokens with their 2-D rotary positions.
//
// The helper and the engine share the GPU and must not compute at the same time (upstream reproduced a request
// stuck for good when they did), so a request with images holds the engine's mutex from the first encode to the
// end of its generation.  Files live in /dev/shm (RAM, no disk), named without spaces: the protocol splits at them.

std::string fnv1a_hex(const std::string& s) {
    unsigned long long h = 1469598103934665603ull;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    char b[24];
    std::snprintf(b, sizeof b, "%016llx", h);
    return b;
}

bool base64_decode(const std::string& in, std::string& out) {
    static const signed char T[256] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,62,-1,63, 52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14, 15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,63,
        -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40, 41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    out.clear();
    out.reserve(in.size() / 4 * 3);
    unsigned acc = 0;
    int bits = 0;
    for (unsigned char c : in) {
        if (c == '=' ) break;
        if (c == '\n' || c == '\r' || c == ' ') continue;
        const signed char v = T[c];
        if (v < 0) return false;
        acc = (acc << 6) | (unsigned) v;
        bits += 6;
        if (bits >= 8) { bits -= 8; out.push_back((char) ((acc >> bits) & 0xFF)); }
    }
    return true;
}

/// The file suffix stb_image (the helper's decoder) reads, from the bytes themselves; "" = not a supported picture.
std::string image_kind(const std::string& bytes) {
    if (bytes.size() >= 8 && bytes.compare(0, 8, "\x89PNG\r\n\x1a\n", 8) == 0) return "png";
    if (bytes.size() >= 3 && (unsigned char) bytes[0] == 0xFF && (unsigned char) bytes[1] == 0xD8) return "jpg";
    if (bytes.size() >= 6 && (bytes.compare(0, 6, "GIF87a") == 0 || bytes.compare(0, 6, "GIF89a") == 0)) return "gif";
    if (bytes.size() >= 2 && bytes[0] == 'B' && bytes[1] == 'M') return "bmp";
    return "";
}

struct ImageInput { std::string bytes, kind; };

/// An OpenAI `image_url` part: a data: URL or a local file path.  URLs are not fetched (this binary opens no
/// outgoing connection).  Returns false with `err` set for anything it cannot read.
bool load_image(const std::string& url, ImageInput& img, std::string& err) {
    std::string bytes;
    if (url.rfind("data:", 0) == 0) {
        const size_t comma = url.find(',');
        if (comma == std::string::npos || url.find(";base64", 0) == std::string::npos || url.find(";base64") > comma) {
            err = "an image data: URL must be base64 encoded"; return false;
        }
        if (!base64_decode(url.substr(comma + 1), bytes)) { err = "the image's base64 data is not valid"; return false; }
    } else if (url.rfind("http://", 0) == 0 || url.rfind("https://", 0) == 0) {
        err = "image URLs are not fetched; send the picture as a data: URL or a local file path"; return false;
    } else {
        std::ifstream f(url, std::ios::binary);
        if (!f) { err = "cannot read the image file " + url; return false; }
        bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    const std::string kind = image_kind(bytes);
    if (kind.empty()) { err = "unsupported image format: send JPEG, PNG, GIF or BMP"; return false; }
    if (bytes.size() > 64u << 20) { err = "the image is larger than 64 MB"; return false; }
    img.bytes = std::move(bytes);
    img.kind = kind;
    return true;
}

class Vision {
public:
    bool start(const Options& o, std::string& err) {
        std::string helper = o.vision_helper;
        if (helper.empty()) {
            char self[4096];
            const ssize_t n = ::readlink("/proc/self/exe", self, sizeof self - 1);
            if (n > 0) { self[n] = 0; helper = std::string(self); helper = helper.substr(0, helper.rfind('/') + 1) + "strata-vision"; }
        }
        if (!file_exists(helper)) { err = "the vision helper is missing: " + helper + " (build tools/vision)"; return false; }
        char d[64];
        std::snprintf(d, sizeof d, "/dev/shm/unbiased-strata-vision-%d", (int) ::getpid());
        dir_ = d;
        if (::mkdir(dir_.c_str(), 0700) != 0 && errno != EEXIST) {
            dir_ = "/tmp/unbiased-strata-vision-" + std::to_string((int) ::getpid());
            if (::mkdir(dir_.c_str(), 0700) != 0 && errno != EEXIST) { err = "cannot create the encoder's directory"; return false; }
        }
        std::vector<std::string> args = {helper, "--mmproj", o.vision_mmproj, "--model", o.model, "--gpu",
                                         "--max-tokens", std::to_string(o.vision_tokens)};
        if (o.vision_threads > 0) { args.push_back("--threads"); args.push_back(std::to_string(o.vision_threads)); }
        int to_child[2], from_child[2];
        if (::pipe(to_child) != 0 || ::pipe(from_child) != 0) { err = "pipe failed"; return false; }
        pid_ = ::fork();
        if (pid_ < 0) { err = "fork failed"; return false; }
        if (pid_ == 0) {
            ::dup2(to_child[0], STDIN_FILENO);
            ::dup2(from_child[1], STDOUT_FILENO);
            ::close(to_child[0]); ::close(to_child[1]); ::close(from_child[0]); ::close(from_child[1]);
            if (::chdir(dir_.c_str()) != 0) ::_exit(126);
            std::vector<char*> argv;
            for (const std::string& a : args) argv.push_back(const_cast<char*>(a.c_str()));
            argv.push_back(nullptr);
            ::execv(argv[0], argv.data());
            std::fprintf(stderr, "unbiased-strata: could not start the vision helper: %s\n", std::strerror(errno));
            ::_exit(127);
        }
        ::close(to_child[0]);
        ::close(from_child[1]);
        in_fd_ = to_child[1];
        out_ = ::fdopen(from_child[0], "r");
        std::string line;
        while (read_line(line)) {
            if (line.rfind("READY", 0) == 0) { n_embd_ = std::atoi(line.c_str() + 5); return n_embd_ > 0; }
            if (line.rfind("ERR", 0) == 0) { err = "vision helper: " + line.substr(3); return false; }
        }
        err = "the vision helper exited before it was ready";
        return false;
    }

    /// The record file (in dir()) and the number of image tokens for a picture; the same bytes encode once.
    bool encode(const ImageInput& img, std::string& record, int& n_tokens, std::string& err) {
        const std::string key = fnv1a_hex(img.bytes) + std::to_string(img.bytes.size());
        for (auto it = cache_.begin(); it != cache_.end(); ++it)
            if (it->key == key) { record = it->record; n_tokens = it->n_tokens; cache_.splice(cache_.begin(), cache_, it); return true; }
        if (!alive_) { err = "the vision helper is not running"; return false; }
        const std::string in_name = "img-" + key + "." + img.kind, out_name = "emb-" + key + ".sve";
        {
            std::ofstream f(dir_ + "/" + in_name, std::ios::binary);
            f.write(img.bytes.data(), (std::streamsize) img.bytes.size());
            if (!f) { err = "cannot write the image for the encoder"; return false; }
        }
        const std::string req = "ENC " + in_name + " " + out_name + "\n";
        bool ok = true;
        size_t off = 0;
        while (ok && off < req.size()) {
            const ssize_t n = ::write(in_fd_, req.data() + off, req.size() - off);
            if (n <= 0) ok = false; else off += (size_t) n;
        }
        std::string line;
        if (!ok || !read_line(line)) { alive_ = false; ::unlink((dir_ + "/" + in_name).c_str()); err = "the vision helper died"; return false; }
        ::unlink((dir_ + "/" + in_name).c_str());
        if (line.rfind("OK ", 0) != 0) { err = "vision helper: " + (line.size() > 4 ? line.substr(4) : line); return false; }
        n_tokens = std::atoi(line.c_str() + 3);
        if (n_tokens < 1) { err = "the encoder returned no image tokens"; return false; }
        record = dir_ + "/" + out_name;
        cache_.push_front({key, record, n_tokens});
        while (cache_.size() > kCache) { ::unlink(cache_.back().record.c_str()); cache_.pop_back(); }
        return true;
    }

    const std::string& dir() const { return dir_; }
    int n_embd() const { return n_embd_; }
    bool alive() const { return alive_; }

    void quit() {
        if (pid_ > 0) {
            const char q[] = "QUIT\n";
            if (::write(in_fd_, q, sizeof q - 1) < 0) { /* already gone */ }
            int status = 0;
            for (int i = 0; i < 50 && ::waitpid(pid_, &status, WNOHANG) == 0; ++i) ::usleep(100 * 1000);
            ::kill(pid_, SIGKILL);
            ::waitpid(pid_, &status, 0);
            pid_ = -1;
        }
        for (const Entry& e : cache_) ::unlink(e.record.c_str());
        cache_.clear();
        if (!dir_.empty()) ::rmdir(dir_.c_str());
    }

private:
    struct Entry { std::string key, record; int n_tokens; };
    static constexpr size_t kCache = 8;   // pictures kept encoded (a chat re-sends its pictures every turn)

    bool read_line(std::string& line) {
        char* buf = nullptr;
        size_t cap = 0;
        const ssize_t n = ::getline(&buf, &cap, out_);
        if (n < 0) { std::free(buf); alive_ = false; return false; }
        line.assign(buf, (size_t) n);
        std::free(buf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        return true;
    }

    pid_t pid_ = -1;
    int in_fd_ = -1;
    FILE* out_ = nullptr;
    std::string dir_;
    int n_embd_ = 0;
    std::atomic<bool> alive_{true};
    std::list<Entry> cache_;
};

// Placeholders that stand in the rendered prompt for things the tokenizer must not see as text: an image
// (U+F0E08 .. U+F0E09 around its index) and a literal vision tag typed by the user (U+F0E0A .. U+F0E09).
const char kImgOpen[] = "\xF3\xB0\xB8\x88", kLitOpen[] = "\xF3\xB0\xB8\x8A", kClose[] = "\xF3\xB0\xB8\x89";
const char* const kVisionLiterals[] = {"<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<|video_pad|>"};

/// User text with any literal vision tag replaced by its placeholder (#554 upstream: typed as text, those strings
/// became the special tokens and took a picture's place).
std::string protect_literals(const std::string& text) {
    std::string out = text;
    for (size_t k = 0; k < sizeof kVisionLiterals / sizeof *kVisionLiterals; ++k) {
        const std::string tag = kVisionLiterals[k], mark = std::string(kLitOpen) + std::to_string(k) + kClose;
        for (size_t at = out.find(tag); at != std::string::npos; at = out.find(tag, at + mark.size())) out.replace(at, tag.size(), mark);
    }
    return out;
}

// ------------------------------------------------------------------------------------------------ text handling

/// The length of the longest prefix of `s` that does not end inside a UTF-8 character.
size_t utf8_complete_prefix(const std::string& s) {
    size_t i = s.size();
    int back = 0;
    while (i > 0 && back < 4) {
        const unsigned char c = (unsigned char) s[i - 1];
        ++back;
        if ((c & 0xC0) == 0x80) { --i; continue; }             // a continuation byte: keep looking for its lead
        const int need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        return need > back ? i - 1 : s.size();
    }
    return s.size();
}

std::string make_id() {
    static std::mt19937_64 rng{std::random_device{}()};
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    char b[40];
    std::snprintf(b, sizeof b, "chatcmpl-%016llx", (unsigned long long) rng());
    return b;
}

long long now_s() {
    return (long long) std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

void json_error(httplib::Response& res, int status, const std::string& message, const char* type) {
    common_json e = common_json::object();
    e["error"] = common_json::object();
    e["error"]["message"] = message;
    e["error"]["type"] = type;
    res.status = status;
    res.set_content(e.dump_safe(), "application/json");
}

// ------------------------------------------------------------------------------------------------- the request

struct Prepared {
    std::vector<llama_token> ids;
    std::string header;        // "GEN <max_new> key=value ... " - the ids follow ("GENI ... <file> " with images)
    common_chat_parser_params parser;   // how to split the output into reasoning, answer and tool calls
    bool stream = false;
    long long max_new = 0;
    std::vector<ImageInput> images;     // in prompt order; empty for a text request
    std::string embeddings;             // the combined record file the GENI line names (deleted after the request)
};

struct Server {
    Options opt;
    Engine engine;
    Vision vision;                       // started only with --vision
    llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
    common_chat_templates_ptr tmpls;
    llama_token tok_vision_start = -1, tok_image_pad = -1, tok_vision_end = -1;

    /// Whether any message carries an image part (decides whether the request takes the engine's turn early).
    static bool request_has_images(const common_json& body) {
        if (!body.is_object() || !body.contains("messages") || !body.at("messages").is_array()) return false;
        for (const auto& m : body.at("messages")) {
            if (!m.is_object() || !m.contains("content") || !m.at("content").is_array()) continue;
            for (const auto& part : m.at("content"))
                if (part.is_object() && part.contains("type") && part.at("type").is_string()) {
                    const std::string t = part.at("type").get<std::string>();
                    if (t == "image_url" || t == "input_image" || t == "image") return true;
                }
        }
        return false;
    }

    /// The request's messages with every image_url part lifted out (into `images`, prompt order) and replaced by a
    /// text placeholder, and any literal vision tag in text protected.  Returns false with `err` for a bad image.
    bool lift_images(const common_json& messages, common_json& out, std::vector<ImageInput>& images, std::string& err) {
        out = messages;
        if (!out.is_array()) return true;
        for (auto& m : out) {
            if (!m.is_object() || !m.contains("content")) continue;
            auto& c = m.at("content");
            if (c.is_string()) { c = protect_literals(c.get<std::string>()); continue; }
            if (!c.is_array()) continue;
            for (auto& part : c) {
                if (!part.is_object() || !part.contains("type")) continue;
                const std::string type = part.at("type").is_string() ? part.at("type").get<std::string>() : "";
                if (type == "text") {
                    if (part.contains("text") && part.at("text").is_string()) part["text"] = protect_literals(part.at("text").get<std::string>());
                } else if (type == "image_url" || type == "input_image" || type == "image") {
                    std::string url;
                    if (part.contains("image_url")) {
                        const auto& u = part.at("image_url");
                        if (u.is_string()) url = u.get<std::string>();
                        else if (u.is_object() && u.contains("url") && u.at("url").is_string()) url = u.at("url").get<std::string>();
                    } else if (part.contains("url") && part.at("url").is_string()) url = part.at("url").get<std::string>();
                    else if (part.contains("source") && part.at("source").is_object()) {   // Anthropic style base64 block
                        const auto& s = part.at("source");
                        if (s.contains("data") && s.at("data").is_string())
                            url = "data:" + s.value("media_type", std::string("image/png")) + ";base64," + s.at("data").get<std::string>();
                    }
                    if (url.empty()) { err = "an image part needs image_url.url"; return false; }
                    if (opt.vision_mmproj.empty()) { err = "this server was started without --vision, so it cannot read images"; return false; }
                    ImageInput img;
                    if (!load_image(url, img, err)) return false;
                    images.push_back(std::move(img));
                    common_json t = common_json::object();
                    t["type"] = "text";
                    t["text"] = std::string(kImgOpen) + std::to_string(images.size() - 1) + kClose;
                    part = t;
                }
            }
        }
        return true;
    }

    /// The rendered prompt to token ids: text through the tokenizer (its special tokens parsed, as the template
    /// writes them), each image placeholder as <|vision_start|> + n_k x <|image_pad|> + <|vision_end|>, each literal
    /// placeholder as the user's text tokenized without special parsing.
    bool tokenize_prompt(const std::string& prompt, const std::vector<int>& image_tokens, std::vector<llama_token>& ids, std::string& err) {
        ids.clear();
        size_t at = 0;
        size_t used = 0;
        while (at < prompt.size()) {
            const size_t img = prompt.find(kImgOpen, at), lit = prompt.find(kLitOpen, at);
            const size_t next = std::min(img, lit);
            if (next == std::string::npos) {
                const std::vector<llama_token> t = common_tokenize(vocab, prompt.substr(at), false, true);
                ids.insert(ids.end(), t.begin(), t.end());
                break;
            }
            if (next > at) {
                const std::vector<llama_token> t = common_tokenize(vocab, prompt.substr(at, next - at), false, true);
                ids.insert(ids.end(), t.begin(), t.end());
            }
            const size_t close = prompt.find(kClose, next + 4);
            if (close == std::string::npos) { err = "malformed placeholder in the rendered prompt"; return false; }
            const int k = std::atoi(prompt.substr(next + 4, close - next - 4).c_str());
            if (next == img) {
                if (k < 0 || (size_t) k >= image_tokens.size()) { err = "the prompt and its images do not match"; return false; }
                ids.push_back(tok_vision_start);
                ids.insert(ids.end(), (size_t) image_tokens[(size_t) k], tok_image_pad);
                ids.push_back(tok_vision_end);
                ++used;
            } else {
                if (k < 0 || (size_t) k >= sizeof kVisionLiterals / sizeof *kVisionLiterals) { err = "malformed literal placeholder"; return false; }
                const std::vector<llama_token> t = common_tokenize(vocab, kVisionLiterals[k], false, /*parse_special=*/false);
                ids.insert(ids.end(), t.begin(), t.end());
            }
            at = close + 4;
        }
        if (used != image_tokens.size()) { err = "the prompt and its images do not match"; return false; }
        return true;
    }

    /// Parses an OpenAI chat request, renders the model's chat template and tokenizes the result.  With images the
    /// caller holds engine.mu: the pictures are encoded here, on the GPU.
    bool prepare(const common_json& body, Prepared& p, std::string& err) {
        if (!body.is_object() || !body.contains("messages")) { err = "the request needs a \"messages\" array"; return false; }
        common_json messages;
        if (!lift_images(body.at("messages"), messages, p.images, err)) return false;
        common_chat_templates_inputs in;
        in.messages = common_chat_msgs_parse_oaicompat(messages);
        if (body.contains("tools") && body.at("tools").is_array() && body.at("tools").size() > 0) {
            in.tools = common_chat_tools_parse_oaicompat(body.at("tools"));
            if (body.contains("tool_choice") && body.at("tool_choice").is_string())
                in.tool_choice = common_chat_tool_choice_parse_oaicompat(body.at("tool_choice").get<std::string>());
            in.parallel_tool_calls = body.value("parallel_tool_calls", false);
        }
        in.add_generation_prompt = true;
        in.use_jinja = true;
        in.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;   // reasoning goes to "reasoning_content"
        in.enable_thinking = true;
        // thinking: OpenAI's reasoning_effort, or the llama.cpp / vLLM convention chat_template_kwargs.enable_thinking
        if (body.contains("reasoning_effort") && body.at("reasoning_effort").is_string()) {
            const std::string effort = body.at("reasoning_effort").get<std::string>();
            if (effort == "none") in.enable_thinking = false;
            else in.chat_template_kwargs["reasoning_effort"] = common_json(effort).dump();
        }
        if (body.contains("chat_template_kwargs") && body.at("chat_template_kwargs").is_object()) {
            const common_json& kw = body.at("chat_template_kwargs");
            if (kw.contains("enable_thinking") && kw.at("enable_thinking").is_boolean())
                in.enable_thinking = kw.at("enable_thinking").get<bool>();
            if (kw.contains("reasoning_effort") && kw.at("reasoning_effort").is_string())
                in.chat_template_kwargs["reasoning_effort"] = kw.at("reasoning_effort").dump();
        }
        const common_chat_params cp = common_chat_templates_apply(tmpls.get(), in);
        std::vector<int> image_tokens;
        std::vector<std::string> records;
        for (const ImageInput& img : p.images) {
            std::string record;
            int n = 0;
            if (!vision.encode(img, record, n, err)) return false;
            image_tokens.push_back(n);
            records.push_back(record);
        }
        if (!tokenize_prompt(cp.prompt, image_tokens, p.ids, err)) return false;
        if (p.ids.empty()) { err = "the prompt is empty"; return false; }
        if (!p.ids.empty() && !p.images.empty() && p.ids.back() == tok_vision_end) { err = "the prompt cannot end in an image: add a question"; return false; }
        if (records.size() == 1) p.embeddings = records[0];
        else if (records.size() > 1) {   // the engine reads the records one after another, in prompt order
            p.embeddings = vision.dir() + "/req-" + make_id().substr(9) + ".sve";
            std::ofstream out(p.embeddings, std::ios::binary);
            for (const std::string& r : records) { std::ifstream f(r, std::ios::binary); out << f.rdbuf(); }
            if (!out) { err = "cannot write the combined embeddings file"; return false; }
        }

        p.parser = common_chat_parser_params(cp);
        p.parser.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        p.parser.parse_tool_calls = true;
        if (!cp.parser.empty()) p.parser.parser.load(cp.parser);

        const long long n = (long long) p.ids.size();
        if (n + 1 > opt.ctx) {
            err = "the prompt is " + std::to_string(n) + " tokens, which does not fit the context of " +
                  std::to_string(opt.ctx);
            return false;
        }
        long long max_new = opt.ctx - n;
        for (const char* key : {"max_completion_tokens", "max_tokens"})
            if (body.contains(key) && body.at(key).is_number()) {
                const long long want = body.at(key).get<long long>();
                if (want >= 1) max_new = std::min(max_new, want);
                break;
            }
        p.max_new = max_new;
        p.stream = body.value("stream", false);

        // sampling: greedy unless the request sets a temperature above zero
        std::string keys;
        const double temperature = body.contains("temperature") && body.at("temperature").is_number()
                                       ? body.at("temperature").get<double>() : 0.0;
        if (temperature > 0.0) {
            const double top_p = body.contains("top_p") && body.at("top_p").is_number() ? body.at("top_p").get<double>() : 0.95;
            long long top_k = body.contains("top_k") && body.at("top_k").is_number() ? body.at("top_k").get<long long>() : 20;
            top_k = std::max(1LL, std::min(64LL, top_k));       // the engine's sampler takes 1..64
            unsigned long long seed;
            if (body.contains("seed") && body.at("seed").is_number()) seed = (unsigned long long) body.at("seed").get<long long>();
            else seed = std::random_device{}();
            char b[160];
            std::snprintf(b, sizeof b, " temperature=%g top_p=%g top_k=%lld seed=%llu", temperature, top_p, top_k, seed);
            keys += b;
            if (body.contains("min_p") && body.at("min_p").is_number()) {
                std::snprintf(b, sizeof b, " min_p=%g", body.at("min_p").get<double>());
                keys += b;
            }
        }
        for (const auto& kv : {std::pair<const char*, const char*>{"presence_penalty", "penalty_present"},
                               std::pair<const char*, const char*>{"frequency_penalty", "penalty_freq"},
                               std::pair<const char*, const char*>{"repeat_penalty", "penalty_repeat"}})
            if (body.contains(kv.first) && body.at(kv.first).is_number()) {
                char b[64];
                std::snprintf(b, sizeof b, " %s=%g", kv.second, body.at(kv.first).get<double>());
                keys += b;
                if (keys.find("penalty_last_n") == std::string::npos) keys += " penalty_last_n=64";
            }
        p.header = p.images.empty() ? "GEN " + std::to_string(max_new) + keys + " "
                                    : "GENI " + std::to_string(max_new) + keys + " " + p.embeddings + " ";
        return true;
    }

    common_json usage_json(const GenResult& r) const {
        common_json u = common_json::object();
        u["prompt_tokens"] = r.prompt_n;
        u["completion_tokens"] = r.produced;
        u["total_tokens"] = r.prompt_n + r.produced;
        return u;
    }

    common_json timings_json(const GenResult& r) const {
        common_json t = common_json::object();
        t["prompt_n"] = r.prompt_n;
        t["prompt_ms"] = r.prompt_ms;
        t["prompt_per_second"] = r.prompt_ms > 0 ? 1000.0 * (double) r.prompt_n / r.prompt_ms : 0.0;
        t["predicted_n"] = r.produced;
        t["predicted_ms"] = r.decode_ms;
        t["predicted_per_second"] = r.decode_ms > 0 ? 1000.0 * (double) r.produced / r.decode_ms : 0.0;
        t["draft_n"] = r.draft_offered;
        t["draft_n_accepted"] = r.draft_accepted;
        return t;
    }

    /// Runs one request.  The text produced so far is parsed by llama.cpp's chat parser after every token, and
    /// `emit` gets what changed (reasoning, answer text, tool-call pieces); it returns false when the client has
    /// gone.  `final_msg` is the whole parsed message.
    GenResult run(const Prepared& p, common_chat_msg& final_msg,
                  const std::function<bool(const std::vector<common_chat_msg_diff>&)>& emit) {
        std::lock_guard<std::recursive_mutex> lk(engine.mu);
        std::string text, pending;
        common_chat_msg msg;
        std::vector<std::string> call_ids;
        bool client_ok = true;
        auto update = [&](bool partial) {
            common_chat_msg now;
            try {
                now = common_chat_parse(text, partial, p.parser);
            } catch (const std::exception&) {
                if (partial) return;                            // not parseable yet: wait for more text
                now = common_chat_msg();
                now.role = "assistant";
                now.content = text;
            }
            if (now.empty()) return;
            now.set_tool_call_ids(call_ids, [] { return "call_" + make_id().substr(9); });
            const std::vector<common_chat_msg_diff> diffs = common_chat_msg_diff::compute_diffs(msg, now);
            msg = now;
            if (client_ok && !diffs.empty()) client_ok = emit(diffs);
        };
        GenResult r = engine.generate(p.header, p.ids, [&](llama_token id) {
            if (llama_vocab_is_eog(vocab, id)) return client_ok;
            pending += common_token_to_piece(vocab, id, /*special=*/true);
            const size_t n = utf8_complete_prefix(pending);
            if (n > 0) {
                text.append(pending, 0, n);
                pending.erase(0, n);
                update(true);
            }
            return client_ok;
        });
        text += pending;
        update(false);
        final_msg = msg;
        if (final_msg.role.empty()) final_msg.role = "assistant";
        return r;
    }

    static const char* finish_reason(const GenResult& r, const common_chat_msg& msg) {
        if (!msg.tool_calls.empty()) return "tool_calls";
        return r.finish == "stop" ? "stop" : "length";
    }

    void handle_chat(const httplib::Request& req, httplib::Response& res) {
        if (!engine.alive()) return json_error(res, 503, "the engine is not running", "server_error");
        const common_json body = common_json::parse_no_throw(req.body);
        if (body.is_discarded()) return json_error(res, 400, "the request body is not valid JSON", "invalid_request_error");
        auto prep = std::make_shared<Prepared>();
        std::string err;
        // A request with pictures encodes them on the GPU inside prepare(), so it takes the engine's turn first and
        // keeps it (the mutex is recursive) through its generation.  Text requests prepare outside the turn.
        const bool has_images = request_has_images(body);
        std::unique_lock<std::recursive_mutex> turn(engine.mu, std::defer_lock);
        if (has_images) turn.lock();
        try {
            if (!prepare(body, *prep, err)) return json_error(res, 400, err, "invalid_request_error");
        } catch (const std::exception& ex) {
            return json_error(res, 400, ex.what(), "invalid_request_error");
        }
        const std::string id = make_id();
        const long long created = now_s();
        // a combined embeddings file (several pictures) is this request's alone; single-picture records stay cached
        auto cleanup = std::shared_ptr<void>(nullptr, [prep](void*) {
            if (prep->images.size() > 1 && !prep->embeddings.empty()) ::unlink(prep->embeddings.c_str());
        });

        if (!prep->stream) {
            common_chat_msg msg;
            const GenResult r = run(*prep, msg, [](const std::vector<common_chat_msg_diff>&) { return true; });
            if (!r.ok) return json_error(res, 500, r.error, "server_error");
            common_json choice = common_json::object();
            choice["index"] = 0;
            choice["message"] = msg.to_json_oaicompat();
            choice["finish_reason"] = finish_reason(r, msg);
            common_json out = common_json::object();
            out["id"] = id;
            out["object"] = "chat.completion";
            out["created"] = created;
            out["model"] = opt.model_name;
            out["choices"] = common_json::array();
            out["choices"].push_back(choice);
            out["usage"] = usage_json(r);
            out["timings"] = timings_json(r);
            res.set_content(out.dump_safe(), "application/json");
            return;
        }

        res.set_header("Cache-Control", "no-cache");
        res.set_header("X-Accel-Buffering", "no");
        res.set_chunked_content_provider("text/event-stream", [this, prep, id, created, cleanup](size_t, httplib::DataSink& sink) {
            auto chunk = [&](const common_json& delta, const char* finish, const GenResult* done) {
                common_json choice = common_json::object();
                choice["index"] = 0;
                choice["delta"] = delta;
                if (finish) choice["finish_reason"] = finish; else choice["finish_reason"] = nullptr;
                common_json out = common_json::object();
                out["id"] = id;
                out["object"] = "chat.completion.chunk";
                out["created"] = created;
                out["model"] = opt.model_name;
                out["choices"] = common_json::array();
                out["choices"].push_back(choice);
                if (done) { out["usage"] = usage_json(*done); out["timings"] = timings_json(*done); }
                const std::string s = "data: " + out.dump_safe() + "\n\n";
                return sink.write(s.data(), s.size());
            };
            common_json first = common_json::object();
            first["role"] = "assistant";
            first["content"] = "";
            bool ok = chunk(first, nullptr, nullptr);
            common_chat_msg msg;
            const GenResult r = run(*prep, msg, [&](const std::vector<common_chat_msg_diff>& diffs) {
                for (const common_chat_msg_diff& df : diffs) {
                    common_json d = common_json::object();
                    if (!df.reasoning_content_delta.empty()) d["reasoning_content"] = df.reasoning_content_delta;
                    if (!df.content_delta.empty()) d["content"] = df.content_delta;
                    if (df.tool_call_index != std::string::npos) {
                        common_json fn = common_json::object();
                        if (!df.tool_call_delta.name.empty()) fn["name"] = df.tool_call_delta.name;
                        fn["arguments"] = df.tool_call_delta.arguments;
                        common_json tc = common_json::object();
                        tc["index"] = (long long) df.tool_call_index;
                        if (!df.tool_call_delta.id.empty()) { tc["id"] = df.tool_call_delta.id; tc["type"] = "function"; }
                        tc["function"] = fn;
                        d["tool_calls"] = common_json::array();
                        d["tool_calls"].push_back(tc);
                    }
                    if (d.size() == 0) continue;
                    ok = ok && chunk(d, nullptr, nullptr);
                }
                return ok;
            });
            if (ok) {
                if (r.ok) {
                    chunk(common_json::object(), finish_reason(r, msg), &r);
                } else {
                    common_json e = common_json::object();
                    e["error"] = common_json::object();
                    e["error"]["message"] = r.error;
                    e["error"]["type"] = "server_error";
                    const std::string s = "data: " + e.dump_safe() + "\n\n";
                    sink.write(s.data(), s.size());
                }
                static const std::string end = "data: [DONE]\n\n";
                sink.write(end.data(), end.size());
            }
            sink.done();
            return true;
        });
    }
};

Server* g_server = nullptr;
httplib::Server* g_http = nullptr;

void on_signal(int) {
    if (g_http) g_http->stop();
}

void quiet_log(ggml_log_level level, const char* text, void*) {
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
}

bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "unbiased-strata: %s needs a value\n", name); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model") o.model = next("--model");
        else if (a == "--pack") o.pack = next("--pack");
        else if (a == "--mtp") o.mtp = next("--mtp");
        else if (a == "--host") o.host = next("--host");
        else if (a == "--port") o.port = std::atoi(next("--port").c_str());
        else if (a == "--model-name") o.model_name = next("--model-name");
        else if (a == "--ctx") o.ctx = std::atoll(next("--ctx").c_str());
        else if (a == "--threads") o.threads = std::atoi(next("--threads").c_str());
        else if (a == "--vision") o.vision_mmproj = next("--vision");
        else if (a == "--vision-helper") o.vision_helper = next("--vision-helper");
        else if (a == "--vision-tokens") o.vision_tokens = std::atoi(next("--vision-tokens").c_str());
        else if (a == "--vision-threads") o.vision_threads = std::atoi(next("--vision-threads").c_str());
        else if (a == "--") {
            for (++i; i < argc; ++i) {
                const std::string e = argv[i];
                // one GPU only: the engine's multi-GPU and remote-expert modes are not part of this fork
                for (const char* no : {"--gpus", "--layer-split", "--split-device", "--peer-device", "--peer-slots",
                                       "--remote-expert", "--vision", "--batch"})
                    if (e.rfind(no, 0) == 0) {
                        std::fprintf(stderr, "unbiased-strata: %s is not supported (single GPU, text only)\n", e.c_str());
                        std::exit(2);
                    }
                o.extra.push_back(e);
            }
        }
        else if (a == "-h" || a == "--help") return false;
        else { std::fprintf(stderr, "unbiased-strata: unknown option %s\n\n", a.c_str()); return false; }
    }
    return !o.model.empty() && !o.pack.empty() && !o.mtp.empty();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--engine") == 0) return strata_engine_main(argc - 1, argv + 1);

    Server srv;
    Options& o = srv.opt;
    if (!parse_args(argc, argv, o)) { usage(); return 2; }

    // every file the engine needs must already be here: nothing is fetched
    std::string shard2 = o.model;
    const size_t at = shard2.rfind("-00001-of-00002");
    if (at == std::string::npos) {
        std::fprintf(stderr, "unbiased-strata: --model must be the first shard (...-00001-of-00002.gguf)\n");
        return 2;
    }
    shard2.replace(at, 15, "-00002-of-00002");
    const std::string profile = o.pack + "/expert-profile.bin";
    for (const std::string& f : {o.model, shard2, o.pack + "/index.txt", profile, o.mtp + "/experts.bin",
                                 o.mtp + "/draft_vocab.bin"})
        if (!file_exists(f)) { std::fprintf(stderr, "unbiased-strata: missing file: %s\n", f.c_str()); return 2; }
    if (o.ctx < 1024) { std::fprintf(stderr, "unbiased-strata: --ctx must be at least 1024\n"); return 2; }

    // the engine, started first: loading takes minutes, and it must be forked before any thread exists
    std::vector<std::string> eargs = {
        "unbiased-strata", "--engine", "--serve", "--pack", o.pack, "--native", o.model, "--ple-gguf", shard2,
        "--expert-profile", profile, "--expert-cache", "auto", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
        "--mtp", o.mtp, "--max-context", std::to_string(o.ctx), "--ple-io", "ram"};
    if (o.ctx > 8192) { eargs.push_back("--kv"); eargs.push_back("int8"); }
    if (o.threads > 0) { eargs.push_back("--pool-workers"); eargs.push_back(std::to_string(o.threads)); }
    if (!o.vision_mmproj.empty()) eargs.push_back("--vision");   // GENI requests; the encoder is this process's child
    for (const std::string& e : o.extra) eargs.push_back(e);
    std::signal(SIGPIPE, SIG_IGN);
    std::string err;
    if (!srv.engine.start(eargs, err)) { std::fprintf(stderr, "unbiased-strata: %s\n", err.c_str()); return 1; }

    // the tokenizer and the chat template, from the GGUF itself
    llama_log_set(quiet_log, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    srv.model = llama_model_load_from_file(o.model.c_str(), mp);
    if (srv.model == nullptr) {
        std::fprintf(stderr, "unbiased-strata: could not read the tokenizer from %s\n", o.model.c_str());
        srv.engine.quit();
        return 1;
    }
    srv.vocab = llama_model_get_vocab(srv.model);
    srv.tmpls = common_chat_templates_init(srv.model, "");

    std::fprintf(stderr, "unbiased-strata: loading the model (this takes a few minutes)...\n");
    if (!srv.engine.wait_ready(err)) {
        std::fprintf(stderr, "unbiased-strata: %s\n", err.c_str());
        srv.engine.quit();
        return 1;
    }
    if (!o.vision_mmproj.empty()) {
        // the three vision tokens, from the vocabulary itself (the engine's image cells are the pad token's id)
        auto one = [&](const char* tag) -> llama_token {
            const std::vector<llama_token> t = common_tokenize(srv.vocab, tag, false, true);
            return t.size() == 1 ? t[0] : -1;
        };
        srv.tok_vision_start = one("<|vision_start|>");
        srv.tok_image_pad = one("<|image_pad|>");
        srv.tok_vision_end = one("<|vision_end|>");
        if (srv.tok_vision_start < 0 || srv.tok_image_pad < 0 || srv.tok_vision_end < 0) {
            std::fprintf(stderr, "unbiased-strata: this model's vocabulary has no vision tokens\n");
            srv.engine.quit();
            return 1;
        }
        // after the engine is ready: the encoder's warm-up takes the GPU memory the engine left free
        if (!file_exists(o.vision_mmproj)) { std::fprintf(stderr, "unbiased-strata: missing file: %s\n", o.vision_mmproj.c_str()); srv.engine.quit(); return 2; }
        if (!srv.vision.start(o, err)) {
            std::fprintf(stderr, "unbiased-strata: %s\n", err.c_str());
            srv.engine.quit();
            return 1;
        }
        std::fprintf(stderr, "unbiased-strata: vision encoder ready (%d-wide embeddings, up to %d tokens per picture)\n",
                     srv.vision.n_embd(), o.vision_tokens);
    }

    httplib::Server http;
    g_server = &srv;
    g_http = &http;
    http.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        res.status = srv.engine.alive() ? 200 : 503;
        res.set_content(srv.engine.alive() ? "{\"status\":\"ok\"}" : "{\"status\":\"error\"}", "application/json");
    });
    http.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        common_json m = common_json::object();
        m["id"] = srv.opt.model_name;
        m["object"] = "model";
        m["owned_by"] = "local";
        m["context_length"] = srv.opt.ctx;
        common_json out = common_json::object();
        out["object"] = "list";
        out["data"] = common_json::array();
        out["data"].push_back(m);
        res.set_content(out.dump_safe(), "application/json");
    });
    http.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        srv.handle_chat(req, res);
    });
    http.set_read_timeout(600, 0);
    http.set_write_timeout(600, 0);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    if (!http.bind_to_port(o.host, o.port)) {
        std::fprintf(stderr, "unbiased-strata: could not listen on %s:%d\n", o.host.c_str(), o.port);
        srv.engine.quit();
        return 1;
    }
    std::fprintf(stderr, "unbiased-strata: ready on http://%s:%d (context %lld tokens)\n", o.host.c_str(), o.port, o.ctx);
    http.listen_after_bind();

    std::fprintf(stderr, "unbiased-strata: stopping\n");
    srv.vision.quit();
    srv.engine.quit();
    llama_model_free(srv.model);
    return 0;
}
