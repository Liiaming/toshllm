// Router trace for residency studies: top-k ids and their weights per (token, layer).
// usage: trace2 MODEL WORKLOAD.txt OUT_PREFIX NGEN [NCMOE]
// The workload file holds one or more turns separated by a line "=====". Turns run in the
// same context, so topic shifts inside a conversation are captured.
// Writes OUT.ids (phase step layer id...), OUT.w (same rows, weights) and OUT.txt (the text).
// Prefill rows carry -(chunk + 1) as their step, so each 512-token chunk can be told apart.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// TRACE_VERIFY: the ids read from the strided top-k view must be the top-k of one of the router's
// contiguous score tensors, row by row; a wrong stride matches none of them
static const char * k_scores[] = { "ffn_moe_probs_masked-", "ffn_moe_probs_biased-", "ffn_moe_probs-", "ffn_moe_logits_biased-", "ffn_moe_logits-" };

struct tctx {
    std::map<std::string, std::vector<float>> scores;   // name -> [ntok][E] of the current ubatch
    long rows_checked = 0, rows_bad = 0;
    long verify_rows = getenv("TRACE_VERIFY") ? atol(getenv("TRACE_VERIFY")) : 0;   // rows to check
    bool verify = false;
    bool weights = getenv("TRACE_WEIGHTS") != nullptr;
    FILE * ids = nullptr, * w = nullptr;
    int step = -1, phase = 0, chunk = 0;
};

static bool cb_eval(ggml_tensor * t, bool ask, void * ud) {
    auto * c = (tctx *) ud;
    const bool is_ids = strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    c->verify = c->rows_checked < c->verify_rows;
    const bool is_w   = c->weights && strncmp(t->name, "ffn_moe_weights-", 16) == 0;
    bool is_score = false;
    for (const char * k : k_scores) is_score = is_score || strncmp(t->name, k, strlen(k)) == 0;
    is_score = is_score && c->verify && t->type == GGML_TYPE_F32;
    if (!is_ids && !is_w && !is_score) return false;
    if (ask) return true;
    if (is_score) {
        std::vector<float> b(t->ne[0] * t->ne[1]);
        for (int64_t j = 0; j < t->ne[1]; j++) ggml_backend_tensor_get(t, b.data() + j * t->ne[0], j * t->nb[1], t->ne[0] * 4);
        c->scores[t->name] = std::move(b);
        return true;
    }
    const int il = atoi(strchr(t->name, '-') + 1);
    const int64_t k = t->ne[0], ntok = t->ne[1];
    if (is_ids) {
        std::vector<int32_t> b(k * ntok);
        if (getenv("TRACE_NEGATIVE_CONTROL")) {
            // the old bug on purpose: the view read as if it were contiguous
            ggml_backend_tensor_get(t, b.data(), 0, std::min<size_t>(k * ntok * 4, ggml_nbytes(t)));
        } else {
            for (int64_t j = 0; j < ntok; j++) ggml_backend_tensor_get(t, b.data() + j * k, j * t->nb[1], k * 4);
        }
        if (c->verify) {
            const std::string suffix = strchr(t->name, '-');
            bool any = false;
            for (const char * kn : k_scores) {
                auto it = c->scores.find(std::string(kn, strlen(kn) - 1) + suffix);
                if (it == c->scores.end()) continue;
                const int64_t E = (int64_t) it->second.size() / ntok;
                bool all = true;
                for (int64_t j = 0; j < ntok && all; j++) {
                    const float * sc = it->second.data() + j * E;
                    std::vector<int> idx(E);
                    for (int e = 0; e < E; e++) idx[e] = e;
                    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [sc](int a, int b2) { return sc[a] > sc[b2]; });
                    std::vector<int> want(idx.begin(), idx.begin() + k), got(b.begin() + j * k, b.begin() + (j + 1) * k);
                    std::sort(want.begin(), want.end()); std::sort(got.begin(), got.end());
                    // a tie at the k-th score can pick either expert: compare the scores instead
                    for (int i = 0; i < k && all; i++) all = want[i] == got[i] || sc[want[i]] == sc[got[i]];
                }
                any = any || all;
            }
            c->rows_checked += ntok;
            if (!any) c->rows_bad += ntok;
        }
        for (int64_t j = 0; j < ntok; j++) {
            fprintf(c->ids, "%d %d %d", c->phase, c->phase ? c->step : -(c->chunk + 1), il);
            for (int64_t i = 0; i < k; i++) fprintf(c->ids, " %d", b[j * k + i]);
            fputc('\n', c->ids);
        }
    } else {
        std::vector<float> b(k * ntok);
        for (int64_t j = 0; j < ntok; j++) ggml_backend_tensor_get(t, b.data() + j * k, j * t->nb[1], k * 4);
        for (int64_t j = 0; j < ntok; j++) {
            fprintf(c->w, "%d %d %d", c->phase, c->phase ? c->step : -(c->chunk + 1), il);
            for (int64_t i = 0; i < k; i++) fprintf(c->w, " %.4g", b[j * k + i]);
            fputc('\n', c->w);
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: trace2 MODEL WORKLOAD OUT_PREFIX NGEN [NCMOE]\n"); return 1; }
    const std::string out = argv[3];
    const int ngen = atoi(argv[4]);
    const int ncmoe = argc > 5 ? atoi(argv[5]) : 0;

    std::ifstream wf(argv[2]);
    std::vector<std::string> turns(1);
    for (std::string ln; std::getline(wf, ln);) {
        if (ln == "=====") { turns.emplace_back(); continue; }
        turns.back() += ln + "\n";
    }

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.load_mode = LLAMA_LOAD_MODE_MLOCK;
    std::vector<llama_model_tensor_buft_override> ov;
    static std::vector<std::string> pats;
    if (ncmoe > 0) {
        for (int i = 0; i < ncmoe; i++)
            pats.push_back("blk\\." + std::to_string(i) + "\\.ffn_(up|down|gate|gate_up)_(ch|)exps");
        for (auto & p : pats) ov.push_back({ p.c_str(), ggml_backend_cpu_buffer_type() });
        ov.push_back({ nullptr, nullptr });
        mp.tensor_buft_overrides = ov.data();
    }
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    tctx tc;
    tc.ids = fopen((out + ".ids").c_str(), "w");
    tc.w   = fopen((out + ".w").c_str(), "w");
    FILE * txt = fopen((out + ".txt").c_str(), "w");

    auto cp = llama_context_default_params();
    cp.n_ctx = getenv("TRACE_CTX") ? atoi(getenv("TRACE_CTX")) : 32768; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.cb_eval = cb_eval; cp.cb_eval_user_data = &tc;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    auto sp = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(1234));

    const char * tmpl = llama_model_chat_template(model, nullptr);
    std::vector<llama_chat_message> hist;
    std::vector<std::string> keep;
    int step = 0, prev_len = 0;
    for (size_t ti = 0; ti < turns.size(); ti++) {
        keep.push_back(turns[ti]);
        hist.push_back({ "user", keep.back().c_str() });
        std::vector<char> buf(1 << 20);
        int n = llama_chat_apply_template(tmpl, hist.data(), hist.size(), true, buf.data(), buf.size());
        std::string full;
        if (n < 0 || n > (int) buf.size()) {
            // a template that needs jinja: the Gemma 4 turn markers
            for (auto & m : hist) full += std::string("<|turn>") + (strcmp(m.role, "user") == 0 ? "user" : "model") + "\n" + m.content + "<turn|>\n";
            full += "<|turn>model\n";
        } else {
            full.assign(buf.data(), n);
        }
        std::string delta = full.substr(prev_len);
        std::vector<llama_token> toks(delta.size() + 16);
        int nt = llama_tokenize(vocab, delta.c_str(), delta.size(), toks.data(), toks.size(), ti == 0, true);
        toks.resize(nt);
        fprintf(stderr, "turn %zu: %d prompt tokens\n", ti, nt);
        fprintf(txt, "\n===== TURN %zu (%d prompt tokens)\n", ti, nt);
        tc.phase = 0;
        for (int i = 0; i < nt; i += 512) {
            tc.chunk++;
            llama_batch b = llama_batch_get_one(toks.data() + i, std::min(512, nt - i));
            if (llama_decode(ctx, b)) { fprintf(stderr, "prefill failed\n"); return 1; }
        }
        tc.phase = 1;
        std::string reply;
        for (int s = 0; s < ngen; s++) {
            llama_token cur = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, cur)) break;
            char piece[256];
            int pn = llama_token_to_piece(vocab, cur, piece, sizeof piece, 0, true);
            if (pn > 0) { reply.append(piece, pn); fwrite(piece, 1, pn, txt); }
            tc.step = step++;
            llama_batch b1 = llama_batch_get_one(&cur, 1);
            if (llama_decode(ctx, b1)) { fprintf(stderr, "decode failed\n"); break; }
        }
        fflush(txt);
        keep.push_back(reply);
        hist.push_back({ "assistant", keep.back().c_str() });
        n = llama_chat_apply_template(tmpl, hist.data(), hist.size(), false, buf.data(), buf.size());
        if (n < 0) {
            std::string all;
            for (auto & m : hist) all += std::string("<|turn>") + (strcmp(m.role, "user") == 0 ? "user" : "model") + "\n" + m.content + "<turn|>\n";
            n = (int) all.size();
        }
        prev_len = n;
    }
    if (tc.verify_rows > 0) fprintf(stderr, "trace verify: %ld rows checked, %ld not the top-k of any router score\n", tc.rows_checked, tc.rows_bad);
    fclose(tc.ids); fclose(tc.w); fclose(txt);
    llama_sampler_free(smpl);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    fprintf(stderr, "done, %d decode steps\n", step);
    return 0;
}
