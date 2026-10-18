// Standalone experiment. Real GGUF slices, frozen CPU grouping, independent
// guarded outputs.
#include "strata/prefill/moe_fused_iq_lab.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <vector>
namespace f = strata::prefill::fused;
void ck(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
struct Buffer {
  uint8_t *base = nullptr, *p = nullptr;
  size_t size;
  explicit Buffer(size_t n) : size(std::max(n, size_t(16))) {
    ck(cudaMalloc(&base, size + 512));
    p = base + 256;
    reset();
  }
  ~Buffer() { cudaFree(base); }
  Buffer(const Buffer &) = delete;
  void reset() { ck(cudaMemset(base, 0xa5, size + 512)); }
  void put(const void *h, size_t n) {
    ck(cudaMemcpy(p, h, n, cudaMemcpyHostToDevice));
  }
  std::vector<uint8_t> get() {
    std::vector<uint8_t> h(size);
    ck(cudaMemcpy(h.data(), p, size, cudaMemcpyDeviceToHost));
    return h;
  }
  void guards() {
    std::array<uint8_t, 512> h;
    ck(cudaMemcpy(h.data(), base, 256, cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(h.data() + 256, p + size, 256, cudaMemcpyDeviceToHost));
    for (auto x : h)
      if (x != 0xa5)
        throw std::runtime_error("buffer guard overwritten");
  }
};
struct Layer {
  int id, gu, down;
  size_t blob, gurow, drow;
  uint64_t go, uo, doo;
  std::string shard;
  f::NativeGeom geom() const {
    return {gu, down, gurow, drow, 640 * gurow, 1280 * gurow};
  }
};
int block(int t) {
  switch (t) {
  case 16:
    return 66;
  case 17:
    return 74;
  case 18:
    return 98;
  case 20:
    return 18;
  case 21:
    return 110;
  case 22:
    return 82;
  case 23:
    return 136;
  case 42:
    return 18;
  default:
    throw std::runtime_error("unsupported real type");
  }
}
std::vector<Layer> layers(const std::string &pack) {
  std::ifstream in(pack + "/native_experts.txt");
  if (!in)
    throw std::runtime_error("layout missing");
  std::vector<Layer> a;
  std::string s;
  while (std::getline(in, s)) {
    if (s.empty() || s[0] == '#')
      continue;
    std::istringstream q(s);
    Layer l{};
    uint64_t offset;
    q >> l.id >> l.gu >> l.down >> offset >> l.blob >> l.go >> l.uo >> l.doo;
    q >> l.shard;
    l.gurow = 10 * block(l.gu);
    l.drow = l.down == 42 ? 10 * 18 : 20 * 18;
    if (l.blob != 1280 * l.gurow + 2560 * l.drow)
      throw std::runtime_error("layout size");
    a.push_back(l);
  }
  if (a.size() != 48)
    throw std::runtime_error("need all 48 layers");
  return a;
}
void read_at(int fd, uint8_t *p, size_t n, uint64_t at) {
  while (n) {
    auto r = pread(fd, p, n, at);
    if (r <= 0)
      throw std::runtime_error("packed read failed");
    p += r;
    at += r;
    n -= r;
  }
}
struct Weights {
  Buffer buf;
  size_t stride;
  int count;
  Weights(const Layer &l, const std::string &model, int c)
      : buf(c * (l.blob + 256)), stride(l.blob + 256), count(c) {
    std::filesystem::path path(model);
    if (!l.shard.empty())
      path = path.parent_path() / l.shard;
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0)
      throw std::runtime_error("model open failed");
    std::vector<uint8_t> h(l.blob);
    for (int e = 0; e < c; e++) {
      read_at(fd, h.data(), 640 * l.gurow, l.go + e * 640 * l.gurow);
      read_at(fd, h.data() + 640 * l.gurow, 640 * l.gurow,
              l.uo + e * 640 * l.gurow);
      read_at(fd, h.data() + 1280 * l.gurow, 2560 * l.drow,
              l.doo + e * 2560 * l.drow);
      ck(cudaMemcpy(buf.p + e * stride + 2, h.data(), h.size(),
                    cudaMemcpyHostToDevice));
    }
    close(fd);
  }
  f::Batch batch(int first, int end) {
    f::Batch b;
    b.e0 = first;
    b.e1 = end;
    for (int e = first; e < end; e++)
      b.blob[e - first] = buf.p + (e % count) * stride + 2;
    return b;
  }
  void guards() {
    buf.guards();
    auto h = buf.get();
    for (int e = 0; e < count; e++) {
      if (h[e * stride] != 0xa5 || h[e * stride + 1] != 0xa5)
        throw std::runtime_error("weight prefix guard");
      for (size_t j = stride - 254; j < stride; j++)
        if (h[e * stride + j] != 0xa5)
          throw std::runtime_error("weight suffix guard");
    }
  }
};
struct Case {
  int n = 0, tokens = 37;
  std::vector<int> counts, offsets, slots;
  Buffer scratch, src, x, xa, h0, h1, d0, d1;
  static int rows(const std::vector<int> &c) {
    return std::accumulate(c.begin(), c.end(), 0);
  }
  Case(std::vector<int> c, bool zero, int ntokens = 37)
      : n(rows(c)), tokens(ntokens), counts(c), offsets(513), slots(n),
        scratch(f::group_bytes(n, 512)), src(n * 4), x(tokens * 2560 * 4),
        xa(f::act_bytes(tokens, 2560)), h0(f::act_bytes(n, 640)),
        h1(f::act_bytes(n, 640)), d0(size_t(n) * 2560 * 4),
        d1(size_t(n) * 2560 * 4) {
    auto sh = std::vector<uint8_t>(scratch.size, 0);
    auto p = reinterpret_cast<int32_t *>(sh.data());
    auto off = p + 512;
    auto fill = p + 1025;
    auto ts = p + 1537;
    size_t ta = ((4 * 512 + 2) * 4 + 15) / 16 * 16;
    auto tl = reinterpret_cast<int2 *>(sh.data() + ta);
    std::vector<int32_t> sources(n);
    int nt = 0;
    for (int e = 0; e < 512; e++) {
      p[e] = counts[e];
      off[e] = offsets[e];
      fill[e] = counts[e];
      ts[e] = nt;
      offsets[e + 1] = offsets[e] + counts[e];
      for (int r = 0; r < counts[e]; r += 64)
        tl[nt++] = make_int2(e, offsets[e] + r);
    }
    off[512] = n;
    ts[512] = nt;
    // Fixed shuffled grouping maps and repeated source tokens. The same tables
    // serve every mode.
    std::iota(slots.begin(), slots.end(), 0);
    std::mt19937 rng(20261011);
    std::shuffle(slots.begin(), slots.end(), rng);
    for (int i = 0; i < n; i++)
      sources[i] = (i * 17 + 3) % tokens;
    scratch.put(sh.data(), sh.size());
    src.put(sources.data(), sources.size() * 4);
    std::vector<float> act(tokens * 2560);
    for (size_t i = 0; i < act.size(); i++)
      act[i] = zero ? 0.f
                    : float(int(rng() % 20001) - 10000) *
                          (.00001f * float(1 << (i % 7)));
    x.put(act.data(), act.size() * 4);
    f::quantize_act_native((float *)x.p, tokens, 2560, xa.p, nullptr);
    ck(cudaDeviceSynchronize());
  }
  void run(Weights &w, const Layer &l, int mode, int tile, int phase, int batch,
           int begin = 0, int end = 512, bool candidate = true) {
    for (int e = begin; e < end; e += batch) {
      auto b = w.batch(e, std::min(e + batch, end));
      f::experts_native_lab(b, l.geom(), 512, n, scratch.p, xa.p,
                            (int32_t *)src.p, candidate ? h1.p : h0.p,
                            (float *)(candidate ? d1.p : d0.p), nullptr, phase,
                            mode, tile);
    }
  }
  void parity(Weights &w, const Layer &l, int mode, int tile, int batch,
              int begin = 0, int end = 512) {
    h0.reset();
    h1.reset();
    d0.reset();
    d1.reset();
    run(w, l, 0, tile, 0, batch, begin, end, false);
    run(w, l, mode, tile, 0, batch, begin, end, true);
    ck(cudaDeviceSynchronize());
    auto a = h0.get(), b = h1.get(), d = d0.get(), e = d1.get();
    for (int r = 0; r < n; r++)
      for (int k = 0; k < 10; k++)
        if (std::memcmp(a.data() + r * 800 + k * 80,
                        b.data() + r * 800 + k * 80, 72))
          throw std::runtime_error("hidden codes/scales parity row=" +
                                   std::to_string(r));
    if (d != e)
      throw std::runtime_error("FP32 down parity");
    // Restore the frozen grouped rows to original routed-slot order, compare
    // all FP32 bits.
    for (int slot : slots)
      if (std::memcmp(d.data() + size_t(slot) * 10240,
                      e.data() + size_t(slot) * 10240, 10240))
        throw std::runtime_error("restored slot parity");
    for (int r = 0; r < n; r++) {
      int expert = std::upper_bound(offsets.begin(), offsets.end(), r) -
                   offsets.begin() - 1;
      if (expert >= begin && expert < end)
        continue;
      for (int j = 0; j < 800; j++)
        if (a[r * 800 + j] != 0xa5 || b[r * 800 + j] != 0xa5)
          throw std::runtime_error("hidden untouched sentinel");
      for (int j = 0; j < 10240; j++)
        if (d[size_t(r) * 10240 + j] != 0xa5 ||
            e[size_t(r) * 10240 + j] != 0xa5)
          throw std::runtime_error("down untouched sentinel");
    }
    h0.guards();
    h1.guards();
    d0.guards();
    d1.guards();
    scratch.guards();
    src.guards();
    x.guards();
    xa.guards();
    w.guards();
    std::cout << "{\"kind\":\"parity\",\"layer\":" << l.id
              << ",\"variant\":" << mode << ",\"tile\":" << tile
              << ",\"batch\":" << batch << ",\"begin\":" << begin
              << ",\"end\":" << end << ",\"rows\":" << n << ",\"passed\":true}"
              << std::endl;
  }
};
void info(const f::NativeLabInfo &i, int layer, int rowclass) {
  std::cout << "{\"kind\":\"resource\",\"layer\":" << layer
            << ",\"class\":" << rowclass << ",\"variant\":" << i.variant
            << ",\"requested_variant\":" << i.requested_variant
            << ",\"phase\":" << i.phase << ",\"format\":" << i.format
            << ",\"threads\":" << i.threads
            << ",\"stages\":" << i.activation_stages
            << ",\"prefetch\":" << i.prefetch
            << ",\"registers\":" << i.registers
            << ",\"occupancy\":" << i.occupancy
            << ",\"shared\":" << i.dynamic_shared << ",\"grid\":" << i.grid
            << ",\"tile\":" << i.tile << "}" << std::endl;
}
std::string query(const char *command) {
  FILE *pipe = popen(command, "r");
  if (!pipe)
    throw std::runtime_error("GPU preflight command failed");
  std::string out;
  char line[1024];
  while (fgets(line, sizeof line, pipe))
    out += line;
  if (pclose(pipe) != 0)
    throw std::runtime_error("GPU preflight query failed");
  return out;
}
void gpu_preflight() {
  if (std::stof(query("nvidia-smi --id=0 --query-gpu=power.limit "
                      "--format=csv,noheader,nounits")) != 100.f)
    throw std::runtime_error("100 W power limit changed before timing");
  std::istringstream processes(
      query("nvidia-smi --id=0 "
            "--query-compute-apps=pid --format=csv,noheader"));
  long pid;
  while (processes >> pid)
    if (pid != getpid())
      throw std::runtime_error("foreign GPU process before timing");
}
float timing(Case &c, Weights &w, const Layer &l, int mode, int phase,
             int iters) {
  cudaEvent_t a, b;
  ck(cudaEventCreate(&a));
  ck(cudaEventCreate(&b));
  ck(cudaEventRecord(a));
  for (int k = 0; k < iters; k++)
    c.run(w, l, mode, 0, phase, 128);
  ck(cudaEventRecord(b));
  ck(cudaEventSynchronize(b));
  float ms;
  ck(cudaEventElapsedTime(&ms, a, b));
  ck(cudaEventDestroy(a));
  ck(cudaEventDestroy(b));
  return ms / iters;
}
int main(int argc, char **argv) {
  try {
    if (argc < 4)
      throw std::runtime_error(
          "usage: native_expert_lab PACK MODEL parity|compact|bench [variant] "
          "[routing.jsonl]");
    auto ls = layers(argv[1]);
    std::string task = argv[3];
    int mode = argc > 4 ? std::stoi(argv[4]) : 1;
    std::array<int, 10> reps{31, 1, 0, 5, 35, 36, 28, 8, 29, 12};
    if (task == "bench") {
      // Routing profiles are real count vectors from frozen-control event
      // profiling. Largest representative per row class.
      if (argc < 6)
        throw std::runtime_error("bench needs observed routing file");
      std::ifstream in(argv[5]);
      if (!in)
        throw std::runtime_error("observed routing file missing");
      std::vector<std::vector<int>> routes(3);
      std::string line;
      while (std::getline(in, line)) {
        std::istringstream s(line);
        int cl;
        s >> cl;
        std::vector<int> c(512);
        for (int &v : c)
          if (!(s >> v))
            throw std::runtime_error("routing vector incomplete");
        if (cl < 0 || cl > 2)
          throw std::runtime_error("routing class");
        if (Case::rows(c) > Case::rows(routes[cl]))
          routes[cl] = c;
      }
      if (std::all_of(routes.begin(), routes.end(),
                      [](const auto &r) { return r.empty(); }))
        throw std::runtime_error("no observed routing vectors");
      for (auto id : reps) {
        auto &l = ls[id];
        Weights w(l, argv[2], 512);
        for (int cl = 0; cl < 3; cl++) {
          if (routes[cl].empty())
            continue;
          Case c(routes[cl], false, std::max(37, Case::rows(routes[cl]) / 10));
          c.parity(w, l, mode, 0, 128);
          for (int phase = 1; phase <= 2; phase++)
            info(f::native_lab_info(w.batch(0, 128), l.geom(), 512, c.n, phase,
                                    mode),
                 id, cl);
          for (int p = 0; p < 3; p++) {
            for (int k = 0; k < 10; k++) {
              c.run(w, l, 0, 0, p, 128);
              c.run(w, l, mode, 0, p, 128);
            }
            ck(cudaDeviceSynchronize());
            for (int pair = 0; pair < 5; pair++)
              for (int order = 0; order < 2; order++) {
                int variant = ((pair + order) % 2) ? mode : 0;
                gpu_preflight();
                auto wall_start =
                    std::chrono::system_clock::now().time_since_epoch();
                auto monotonic_start =
                    std::chrono::steady_clock::now().time_since_epoch();
                float ms = timing(c, w, l, variant, p, 20);
                auto wall_end =
                    std::chrono::system_clock::now().time_since_epoch();
                auto monotonic_end =
                    std::chrono::steady_clock::now().time_since_epoch();
                std::cout
                    << "{\"kind\":\"timing\",\"layer\":" << id
                    << ",\"class\":" << cl << ",\"phase\":" << p
                    << ",\"variant\":" << variant << ",\"pair\":" << pair
                    << ",\"order\":" << order
                    << ",\"iterations\":20,\"ms\":" << ms
                    << ",\"wall_start_ns\":"
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(
                           wall_start)
                           .count()
                    << ",\"wall_end_ns\":"
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(
                           wall_end)
                           .count()
                    << ",\"monotonic_start_ns\":"
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(
                           monotonic_start)
                           .count()
                    << ",\"monotonic_end_ns\":"
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(
                           monotonic_end)
                           .count()
                    << "}" << std::endl;
              }
          }
        }
      }
    } else {
      bool compact = task == "compact";
      bool narrow = compact && argc > 5 && std::string(argv[5]) == "narrow";
      if (task != "parity" && !compact)
        throw std::runtime_error("unknown task");
      std::vector<int> counts(512);
      std::vector<int> edge{0,  1,  7,   8,   15,  16,  31,   32,  63,
                            64, 65, 127, 128, 129, 257, 1025, 2049};
      for (size_t j = 0; j < edge.size(); j++)
        counts[7 + j] = compact ? ((j % 4 == 0)   ? 1
                                   : (j % 4 == 1) ? 17
                                   : (j % 4 == 2) ? 65
                                                  : 129)
                                : edge[j];
      if (narrow) {
        std::fill(counts.begin(), counts.end(), 0);
        counts[7] = 1;
        counts[8] = 65;
      }
      for (auto id : reps) {
        if (narrow && id != 31 && id != 12)
          continue;
        auto &l = ls[id];
        Weights w(l, argv[2], compact ? 1 : 512);
        for (bool zero : {false, true}) {
          if (narrow && zero)
            continue;
          Case c(counts, zero);
          for (int tile : {64, 128}) {
            c.parity(w, l, mode, tile, 17, 7, 24);
            if (!compact)
              for (int batch : {1, 127, 128})
                c.parity(w, l, mode, tile, batch);
          }
        }
        if (!compact) {
          std::vector<int> uniform(512, 1);
          Case c(uniform, false);
          c.parity(w, l, mode, 0, 128);
        }
      }
      if (!compact)
        for (auto &l : ls) {
          Weights w(l, argv[2], 1);
          std::vector<int> c(512);
          c[l.id * 7 % 512] = 65;
          Case sample(c, false);
          sample.parity(w, l, mode, 64, 128);
          sample.parity(w, l, mode, 128, 128);
        }
    }
    std::cout << "{\"kind\":\"complete\",\"task\":\"" << task
              << "\",\"variant\":" << mode << ",\"passed\":true}" << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAILED: " << e.what() << std::endl;
    return 1;
  }
}
