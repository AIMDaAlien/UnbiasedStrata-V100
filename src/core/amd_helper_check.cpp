#include "strata/core/amd_experts.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <signal.h>
#include <unistd.h>

namespace c = strata::core;
namespace cpu = strata::kernels::cpu;
namespace a = strata::amd;
class Source final : public c::ExpertSource {
    strata::GgufModel model;
    std::vector<uint8_t> buffer;
public:
    explicit Source(const char* shard) : model(strata::gguf_split_paths(shard)) {}
    const uint8_t* blob(int64_t l, int64_t e) override {
        const auto& f = cpu::expert_layout().fmt[(size_t)l];
        buffer.resize(f.bytes);
        const char* roles[] = {"gate", "up", "down"};
        const size_t offsets[] = {0, f.up_off, f.down_off, f.bytes};
        for (int r = 0; r < 3; ++r) {
            size_t shard = 0;
            auto* t = model.find("blk."+std::to_string(l)+".ffn_"+roles[r]+"_exps.weight", &shard);
            if (!t) return nullptr;
            const size_t bytes = offsets[r+1]-offsets[r];
            std::memcpy(buffer.data()+offsets[r], model.shard(shard).tensor_data(*t)+(size_t)e*bytes, bytes);
        }
        return buffer.data();
    }
};
static void require(bool ok, const std::string& error) {
    if (!ok) throw std::runtime_error(error);
}
static std::vector<float> reference(Source& src, const std::vector<float>& x, int layer, int tokens, int expert=7) {
    const auto& f = cpu::expert_layout().fmt[layer];
    std::vector<float> result((size_t)tokens*a::H);
    for (int t = 0; t < tokens; ++t) {
        uint8_t act[cpu::kNativeActBytes], h[cpu::kNativeHBytes]; float ff[a::FF];
        cpu::native_quant_act(f, x.data()+(size_t)t*a::H, act);
        const void* ap[] = {act}; float* fp[] = {ff};
        auto* blob = src.blob(layer, expert);
        cpu::native_gu_rows(f, blob, ap, 1, fp, 0, a::FF);
        cpu::native_quant_h(f, ff, h);
        const void* hp[] = {h}; float* op[] = {result.data()+(size_t)t*a::H};
        cpu::native_down_rows(f, blob, hp, 1, op, 0, a::H);
    }
    return result;
}
int main(int argc, char** argv) {
    // Re-exec this test as a fake inherited worker that sends stale RUN results.
    if (argc == 4) {
        const int fd = std::atoi(argv[1]);
        a::Message m;
        while (a::receive(fd, m, -1)) {
            if (m.op == a::Run) --m.sequence;
            if (!a::send(fd, m) || m.op == a::Shutdown) break;
        }
        return 0;
    }
    if (argc != 5) { std::cerr << "usage: helper-check WORKER ROCr-UUID SHARD PACK\n"; return 2; }
    try {
        std::string error;
        require(cpu::expert_layout_load(argv[4], 48, 512, error), error);
        Source source(argv[3]); c::ExpertCache empty;
        std::vector<std::pair<int32_t,int32_t>> pairs{{0,7},{0,8},{0,9},{1,7},{20,7},{47,7}};
        c::AmdExperts helper;
        require(helper.open_worker(argv[1], argv[2], 64ULL<<20, pairs, empty, source, error), error);
        std::mt19937 rng(42); std::normal_distribution<float> normal(0,1);
        for (int nt : {1,4,256}) for (int l : {0,1,20,47}) {
            std::vector<float> x((size_t)nt*a::H), output(x.size());
            for (float& v : x) v = normal(rng);
            std::vector<int32_t> ids(nt,7);
            require(helper.begin(l,x.data(),ids.data(),nt,1,nullptr,nullptr,error),error);
            require(helper.finish(output.data(),error),error);
            const auto ref = reference(source,x,l,nt);
            double diff=0, denom=0;
            for (size_t i=0;i<ref.size();++i) {
                require(std::isfinite(output[i]),"nonfinite AMD result");
                diff += std::abs(output[i]-ref[i]); denom += std::abs(ref[i]);
            }
            require(diff/(denom+1e-30)<0.03,"AMD output differs by >=3% from CPU");
            std::cout << "layer "<<l<<" tokens "<<nt<<" cpu-gpu relative "<<diff/denom<<'\n';
        }
        // Interleaved expert groups and token indices must return to their
        // original routed row, including repeated experts within the batch.
        {
            std::vector<float> x(4*a::H), output(40*a::H);
            for (float& v:x) v=normal(rng);
            std::vector<int32_t> ids(40);
            for (int i=0;i<40;++i) ids[i]=7+i%3;
            require(helper.begin(0,x.data(),ids.data(),4,10,nullptr,nullptr,error),error);
            require(helper.finish(output.data(),error),error);
            for (int e=7;e<=9;++e) {
                const auto ref=reference(source,x,0,4,e);
                double diff=0,denom=0;
                for (int i=0;i<40;++i) if (ids[i]==e) for (int n=0;n<a::H;++n) {
                    diff+=std::abs(output[(size_t)i*a::H+n]-ref[(size_t)(i/10)*a::H+n]);
                    denom+=std::abs(ref[(size_t)(i/10)*a::H+n]);
                }
                require(diff/denom<0.03,"interleaved expert/token result ordering failed");
            }
        }
        std::vector<float> x(a::H,0.25f), output(a::H,-999);
        int32_t id=7;
        require(helper.begin(0,x.data(),&id,1,1,nullptr,nullptr,error),error);
        // The only child belongs to this test; kill it with a request outstanding.
        std::ifstream children("/proc/self/task/"+std::to_string(getpid())+"/children");
        pid_t child=0; children>>child; require(child>0,"worker child missing"); kill(child,SIGKILL);
        require(helper.finish(output.data(),error),error);
        auto ref=reference(source,x,0,1);
        require(output==ref && helper.fallback_rows==1,"worker death did not recompute exact CPU output");
        helper.close_worker();
        require(helper.open_worker(argv[1],argv[2],64ULL<<20,pairs,empty,source,error),error);
        require(helper.begin(0,x.data(),&id,1,1,nullptr,nullptr,error),error);
        helper.cancel();
        std::fill(output.begin(),output.end(),-999);
        require(helper.finish(output.data(),error) && output[0]==-999 && !helper.owns(0),"cancel published results");
        helper.close_worker();
        c::AmdExperts stale;
        require(stale.open_worker(argv[0],argv[2],64ULL<<20,pairs,empty,source,error),error);
        require(stale.begin(0,x.data(),&id,1,1,nullptr,nullptr,error),error);
        require(stale.finish(output.data(),error),error);
        require(output==ref && stale.fallback_rows==1,"stale result was accepted");
        stale.close_worker();
        std::cout << "helper-check: parity, worker death, cancel, stale sequence and shutdown passed\n";
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
