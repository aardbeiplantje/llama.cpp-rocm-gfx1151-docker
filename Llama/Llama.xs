#define PERL_NO_GET_CONTEXT
#include "llama.h"

#include <EXTERN.h>
#include <perl.h>
#include <XSUB.h>

static llama_token* TLlama_tokens = NULL;
static int32_t TLlama_token_count = 0;
static STRLEN TLlama_text_len = 0;
static llama_token* TLlama_loaded_tokens = NULL;
static size_t TLlama_loaded_count = 0;

MODULE = Llama            PACKAGE = Llama

# ============================================================================
# Lifecycle
# ============================================================================

void
llama_backend_init()
    CODE:
        llama_backend_init();

void
llama_backend_free()
    CODE:
        llama_backend_free();

# ============================================================================
# Model loading / freeing
# ============================================================================

IV
llama_model_load_from_file(char* path_model)
    CODE:
        struct llama_model_params params = llama_model_default_params();
        params.n_gpu_layers = 999999;
        RETVAL = (IV)llama_model_load_from_file(path_model, params);
    OUTPUT:
        RETVAL

void
llama_model_free(IV model)
    CODE:
        llama_model_free((struct llama_model*)model);

# ============================================================================
# Model query functions
# ============================================================================

IV
llama_model_n_ctx_train(IV model)
    CODE:
        RETVAL = (IV)llama_model_n_ctx_train((const struct llama_model*)model);
    OUTPUT:
        RETVAL

IV
llama_model_n_embd(IV model)
    CODE:
        RETVAL = (IV)llama_model_n_embd((const struct llama_model*)model);
    OUTPUT:
        RETVAL

IV
llama_model_n_layer(IV model)
    CODE:
        RETVAL = (IV)llama_model_n_layer((const struct llama_model*)model);
    OUTPUT:
        RETVAL

UV
llama_model_n_params(IV model)
    CODE:
        RETVAL = (UV)llama_model_n_params((const struct llama_model*)model);
    OUTPUT:
        RETVAL

UV
llama_model_size(IV model)
    CODE:
        RETVAL = (UV)llama_model_size((const struct llama_model*)model);
    OUTPUT:
        RETVAL

IV
llama_model_desc(IV model, char* buf, IV buf_size)
    CODE:
        RETVAL = (IV)llama_model_desc((const struct llama_model*)model, buf, (size_t)buf_size);
    OUTPUT:
        RETVAL

const char*
llama_model_chat_template(IV model, char* name)
    CODE:
        RETVAL = llama_model_chat_template((const struct llama_model*)model, name);
    OUTPUT:
        RETVAL

# ============================================================================
# Context creation / destruction
# ============================================================================

IV
llama_init_from_model(IV model, IV n_ctx, IV n_batch, IV n_threads, IV n_threads_batch, bool embeddings)
    CODE:
        struct llama_context_params params = llama_context_default_params();
        params.n_ctx = (uint32_t)n_ctx;
        params.n_batch = (uint32_t)n_batch;
        params.n_ubatch = 256;
        params.n_seq_max = 1;
        params.n_threads = (int32_t)n_threads;
        params.n_threads_batch = (int32_t)n_threads_batch;
        params.embeddings = embeddings;
        RETVAL = (IV)llama_init_from_model((struct llama_model*)model, params);
    OUTPUT:
        RETVAL

void
llama_free(IV ctx)
    CODE:
        llama_free((struct llama_context*)ctx);

# ============================================================================
# Context query
# ============================================================================

IV
llama_n_ctx(IV ctx)
    CODE:
        RETVAL = (IV)llama_n_ctx((const struct llama_context*)ctx);
    OUTPUT:
        RETVAL

IV
llama_n_batch(IV ctx)
    CODE:
        RETVAL = (IV)llama_n_batch((const struct llama_context*)ctx);
    OUTPUT:
        RETVAL

IV
llama_n_seq_max(IV ctx)
    CODE:
        RETVAL = (IV)llama_n_seq_max((const struct llama_context*)ctx);
    OUTPUT:
        RETVAL

# ============================================================================
# Batch management
# ============================================================================

IV
llama_batch_init(IV n_tokens, IV embd, IV n_seq_max)
    CODE:
        struct llama_batch batch = llama_batch_init((int32_t)n_tokens, (int32_t)embd, (int32_t)n_seq_max);
        RETVAL = (IV)malloc(sizeof(struct llama_batch));
        *(struct llama_batch*)RETVAL = batch;
    OUTPUT:
        RETVAL

void
llama_batch_free(IV batch)
    CODE:
        llama_batch_free(*(struct llama_batch*)batch);
        free((void*)batch);

# ============================================================================
# Batch field setters
# ============================================================================

void
llama_batch_set_n_tokens(IV batch, IV n_tokens)
    CODE:
        struct llama_batch* b = (struct llama_batch*)batch;
        b->n_tokens = (int32_t)n_tokens;

void
llama_batch_set_token(IV batch, IV idx, IV token, IV pos, SV* seq_id_sv)
    CODE:
        struct llama_batch* b = (struct llama_batch*)batch;
        IV n_seq = 0;
        AV* seq_id_av = NULL;
        if (SvROK(seq_id_sv) && SvTYPE(SvRV(seq_id_sv)) == SVt_PVAV) {
            seq_id_av = (AV*)SvRV(seq_id_sv);
            n_seq = av_len(seq_id_av) + 1;
        }
        b->token[idx] = (llama_token)token;
        b->pos[idx] = (llama_pos)pos;
        b->n_seq_id[idx] = (int32_t)n_seq;
        b->logits[idx] = (int8_t)1;
        if (seq_id_av) {
            for (int32_t i = 0; i < n_seq; i++) {
                SV* sv = *av_fetch(seq_id_av, i, 0);
                b->seq_id[idx][i] = (llama_seq_id)SvIV(sv);
            }
        }

# ============================================================================
# Inference
# ============================================================================

IV
llama_decode(IV ctx, IV batch)
    CODE:
        RETVAL = (IV)llama_decode((struct llama_context*)ctx, *(struct llama_batch*)batch);
    OUTPUT:
        RETVAL

IV
llama_encode(IV ctx, IV batch)
    CODE:
        RETVAL = (IV)llama_encode((struct llama_context*)ctx, *(struct llama_batch*)batch);
    OUTPUT:
        RETVAL

# ============================================================================
# Logits retrieval — returns pointer to float array
# ============================================================================

float*
llama_get_logits(IV ctx)
    CODE:
        RETVAL = llama_get_logits((struct llama_context*)ctx);
    OUTPUT:
        RETVAL

# ============================================================================
# Logits retrieval (ith token) — returns single float
# ============================================================================

NV
llama_get_logits_ith(IV ctx, IV i)
    CODE:
        float* logits = llama_get_logits_ith((struct llama_context*)ctx, (int32_t)i);
        RETVAL = logits ? (NV)logits[0] : 0.0;
    OUTPUT:
        RETVAL

# ============================================================================
# Embeddings retrieval — returns pointer to float array
# ============================================================================

float*
llama_get_embeddings(IV ctx)
    CODE:
        RETVAL = llama_get_embeddings((struct llama_context*)ctx);
    OUTPUT:
        RETVAL

# ============================================================================
# Tokenization — persistent buffer for tokenized results
# ============================================================================

IV
llama_tokenize(IV vocab, SV* text, IV max_tokens, bool add_special, bool parse_special)
    CODE:
        if (TLlama_tokens) free(TLlama_tokens);
        TLlama_text_len = 0;
        char* text_str = SvPV(text, TLlama_text_len);
        TLlama_tokens = (llama_token*)malloc((size_t)max_tokens * sizeof(llama_token));
        int32_t n = llama_tokenize((const struct llama_vocab*)vocab, text_str, (int32_t)TLlama_text_len, TLlama_tokens, (int32_t)max_tokens, add_special, parse_special);
        if (n < 0) {
            free(TLlama_tokens);
            TLlama_tokens = NULL;
            TLlama_token_count = 0;
            RETVAL = -1;
        } else {
            TLlama_token_count = n;
            RETVAL = (IV)n;
        }
    OUTPUT:
        RETVAL

IV
llama_tokenize_get_token(IV idx)
    CODE:
        int32_t i = (int32_t)idx;
        if (!TLlama_tokens || i < 0 || i >= TLlama_token_count) {
            RETVAL = -1;
        } else {
            RETVAL = (IV)TLlama_tokens[i];
        }
    OUTPUT:
        RETVAL

void
llama_tokenize_free()
    CODE:
        free(TLlama_tokens);
        TLlama_tokens = NULL;
        TLlama_token_count = 0;

# ============================================================================
# Token to text piece
# ============================================================================

IV
llama_token_to_piece(IV vocab, IV token, char* buf, IV buf_len, IV lstrip, bool special)
    CODE:
        RETVAL = (IV)llama_token_to_piece((const struct llama_vocab*)vocab, (llama_token)token, buf, (int32_t)buf_len, (int32_t)lstrip, special);
    OUTPUT:
        RETVAL

# ============================================================================
# Detokenization (tokens → text)
# ============================================================================

IV
llama_detokenize(IV vocab, IV* tokens, IV n_tokens, char* buf, IV buf_len, bool remove_special, bool unparse_special)
    CODE:
        RETVAL = (IV)llama_detokenize((const struct llama_vocab*)vocab, (const llama_token*)tokens, (int32_t)n_tokens, buf, (int32_t)buf_len, remove_special, unparse_special);
    OUTPUT:
        RETVAL

# ============================================================================
# Vocabulary query
# ============================================================================

IV
llama_vocab_n_tokens(IV vocab)
    CODE:
        RETVAL = (IV)llama_vocab_n_tokens((const struct llama_vocab*)vocab);
    OUTPUT:
        RETVAL

llama_token
llama_vocab_bos(IV vocab)
    CODE:
        RETVAL = llama_vocab_bos((const struct llama_vocab*)vocab);
    OUTPUT:
        RETVAL

llama_token
llama_vocab_eos(IV vocab)
    CODE:
        RETVAL = llama_vocab_eos((const struct llama_vocab*)vocab);
    OUTPUT:
        RETVAL

llama_token
llama_vocab_eot(IV vocab)
    CODE:
        RETVAL = llama_vocab_eot((const struct llama_vocab*)vocab);
    OUTPUT:
        RETVAL

llama_token
llama_vocab_nl(IV vocab)
    CODE:
        RETVAL = llama_vocab_nl((const struct llama_vocab*)vocab);
    OUTPUT:
        RETVAL

# ============================================================================
# Sampling — individual samplers
# ============================================================================

IV
llama_sampler_init_greedy()
    CODE:
        RETVAL = (IV)llama_sampler_init_greedy();
    OUTPUT:
        RETVAL

IV
llama_sampler_init_top_k(IV k)
    CODE:
        RETVAL = (IV)llama_sampler_init_top_k((int32_t)k);
    OUTPUT:
        RETVAL

IV
llama_sampler_init_top_p(NV p, IV min_keep)
    CODE:
        RETVAL = (IV)llama_sampler_init_top_p((float)p, (size_t)min_keep);
    OUTPUT:
        RETVAL

IV
llama_sampler_init_temp(NV t)
    CODE:
        RETVAL = (IV)llama_sampler_init_temp((float)t);
    OUTPUT:
        RETVAL

IV
llama_sampler_init_dist(IV seed)
    CODE:
        RETVAL = (IV)llama_sampler_init_dist((uint32_t)seed);
    OUTPUT:
        RETVAL

# ============================================================================
# Sampler chain
# ============================================================================

IV
llama_sampler_chain_init()
    CODE:
        struct llama_sampler_chain_params params = llama_sampler_chain_default_params();
        RETVAL = (IV)llama_sampler_chain_init(params);
    OUTPUT:
        RETVAL

void
llama_sampler_chain_add(IV chain, IV sampler)
    CODE:
        llama_sampler_chain_add((struct llama_sampler*)chain, (struct llama_sampler*)sampler);

IV
llama_sampler_chain_get(IV chain, IV i)
    CODE:
        RETVAL = (IV)llama_sampler_chain_get((struct llama_sampler*)chain, (int32_t)i);
    OUTPUT:
        RETVAL

IV
llama_sampler_chain_n(IV chain)
    CODE:
        RETVAL = (IV)llama_sampler_chain_n((const struct llama_sampler*)chain);
    OUTPUT:
        RETVAL

void
llama_sampler_free(IV sampler)
    CODE:
        llama_sampler_free((struct llama_sampler*)sampler);

void
llama_sampler_chain_free(IV chain)
    CODE:
        llama_sampler_free((struct llama_sampler*)chain);

# ============================================================================
# Sampler apply — sample a token from context
# ============================================================================

llama_token
llama_sampler_sample(IV sampler, IV ctx, IV idx)
    CODE:
        RETVAL = llama_sampler_sample((struct llama_sampler*)sampler, (struct llama_context*)ctx, (int32_t)idx);
    OUTPUT:
        RETVAL

void
llama_sampler_reset(IV sampler)
    CODE:
        llama_sampler_reset((struct llama_sampler*)sampler);

# ============================================================================
# Performance info
# ============================================================================

NV
llama_perf_context_load_ms(IV ctx)
    CODE:
        struct llama_perf_context_data data = llama_perf_context((const struct llama_context*)ctx);
        RETVAL = data.t_load_ms;
    OUTPUT:
        RETVAL

NV
llama_perf_context_eval_ms(IV ctx)
    CODE:
        struct llama_perf_context_data data = llama_perf_context((const struct llama_context*)ctx);
        RETVAL = data.t_eval_ms;
    OUTPUT:
        RETVAL

void
llama_perf_context_reset(IV ctx)
    CODE:
        llama_perf_context_reset((struct llama_context*)ctx);

# ============================================================================
# Model → vocab query
# ============================================================================

IV
llama_model_get_vocab(IV model)
    CODE:
        RETVAL = (IV)(IV)llama_model_get_vocab((const struct llama_model*)model);
    OUTPUT:
        RETVAL

# ============================================================================
# Context → model query
# ============================================================================

IV
llama_get_model(IV ctx)
    CODE:
        RETVAL = (IV)(IV)llama_get_model((const struct llama_context*)ctx);
    OUTPUT:
        RETVAL

# ============================================================================
# KV cache state — get size
# ============================================================================

UV
llama_state_get_size(IV ctx)
    CODE:
        RETVAL = (UV)llama_state_get_size((struct llama_context*)ctx);
    OUTPUT:
        RETVAL

# ============================================================================
# KV cache state — get data into packed binary string
# ============================================================================

SV*
llama_state_get_data(IV ctx)
    CODE:
        size_t size = llama_state_get_size((struct llama_context*)ctx);
        uint8_t* buf = (uint8_t*)malloc(size);
        size_t copied = llama_state_get_data((struct llama_context*)ctx, buf, size);
        RETVAL = newSVpv((char*)buf, copied);
        free(buf);
    OUTPUT:
        RETVAL

# ============================================================================
# KV cache state — set data from packed binary string
# ============================================================================

UV
llama_state_set_data(IV ctx, SV* data_sv)
    CODE:
        STRLEN len;
        char* data = SvPV(data_sv, len);
        RETVAL = (UV)llama_state_set_data((struct llama_context*)ctx, (const uint8_t*)data, len);
    OUTPUT:
        RETVAL

# ============================================================================
# KV cache state — save session file
# ============================================================================

bool
llama_state_save_file(IV ctx, SV* path_sv, IV* tokens, IV n_tokens)
    CODE:
        STRLEN path_len;
        char* path = SvPV(path_sv, path_len);
        RETVAL = llama_state_save_file((struct llama_context*)ctx, path, (const llama_token*)tokens, (size_t)n_tokens);
    OUTPUT:
        RETVAL

# ============================================================================
# KV cache state — load session file
# ============================================================================

bool
llama_state_load_file(IV ctx, SV* path_sv, IV capacity)
    CODE:
        STRLEN path_len;
        char* path = SvPV(path_sv, path_len);
        if (TLlama_loaded_tokens) free(TLlama_loaded_tokens);
        TLlama_loaded_tokens = (llama_token*)malloc((size_t)capacity * sizeof(llama_token));
        size_t token_count = 0;
        RETVAL = llama_state_load_file((struct llama_context*)ctx, path, TLlama_loaded_tokens, (size_t)capacity, &token_count);
        TLlama_loaded_count = token_count;
    OUTPUT:
        RETVAL

IV
llama_state_load_file_count()
    CODE:
        RETVAL = (IV)TLlama_loaded_count;
    OUTPUT:
        RETVAL

void
llama_state_load_file_free()
    CODE:
        free(TLlama_loaded_tokens);
        TLlama_loaded_tokens = NULL;
        TLlama_loaded_count = 0;

IV
llama_state_load_file_token(IV idx)
    CODE:
        int32_t i = (int32_t)idx;
        if (!TLlama_loaded_tokens || i < 0 || i >= TLlama_loaded_count) {
            RETVAL = -1;
        } else {
            RETVAL = (IV)TLlama_loaded_tokens[i];
        }
    OUTPUT:
        RETVAL

# ============================================================================
# Per-sequence state — get size
# ============================================================================

UV
llama_state_seq_get_size(IV ctx, IV seq_id)
    CODE:
        RETVAL = (UV)llama_state_seq_get_size((struct llama_context*)ctx, (llama_seq_id)seq_id);
    OUTPUT:
        RETVAL

# ============================================================================
# Per-sequence state — get data
# ============================================================================

SV*
llama_state_seq_get_data(IV ctx, IV seq_id)
    CODE:
        size_t size = llama_state_seq_get_size((struct llama_context*)ctx, (llama_seq_id)seq_id);
        uint8_t* buf = (uint8_t*)malloc(size);
        size_t copied = llama_state_seq_get_data((struct llama_context*)ctx, buf, size, (llama_seq_id)seq_id);
        RETVAL = newSVpv((char*)buf, copied);
        free(buf);
    OUTPUT:
        RETVAL

# ============================================================================
# Per-sequence state — set data
# ============================================================================

UV
llama_state_seq_set_data(IV ctx, SV* data_sv, IV seq_id)
    CODE:
        STRLEN len;
        char* data = SvPV(data_sv, len);
        RETVAL = (UV)llama_state_seq_set_data((struct llama_context*)ctx, (const uint8_t*)data, len, (llama_seq_id)seq_id);
    OUTPUT:
        RETVAL

# ============================================================================
# Per-sequence state — save session file
# ============================================================================

UV
llama_state_seq_save_file(IV ctx, SV* path_sv, IV seq_id, IV* tokens, IV n_tokens)
    CODE:
        STRLEN path_len;
        char* path = SvPV(path_sv, path_len);
        RETVAL = (UV)llama_state_seq_save_file((struct llama_context*)ctx, path, (llama_seq_id)seq_id, (const llama_token*)tokens, (size_t)n_tokens);
    OUTPUT:
        RETVAL

# ============================================================================
# Per-sequence state — load session file
# ============================================================================

bool
llama_state_seq_load_file(IV ctx, SV* path_sv, IV seq_id, IV capacity)
    CODE:
        STRLEN path_len;
        char* path = SvPV(path_sv, path_len);
        if (TLlama_loaded_tokens) free(TLlama_loaded_tokens);
        TLlama_loaded_tokens = (llama_token*)malloc((size_t)capacity * sizeof(llama_token));
        size_t token_count = 0;
        RETVAL = llama_state_seq_load_file((struct llama_context*)ctx, path, (llama_seq_id)seq_id, TLlama_loaded_tokens, (size_t)capacity, &token_count) > 0;
        TLlama_loaded_count = token_count;
    OUTPUT:
        RETVAL

IV
llama_state_seq_load_file_count()
    CODE:
        RETVAL = (IV)TLlama_loaded_count;
    OUTPUT:
        RETVAL

void
llama_state_seq_load_file_free()
    CODE:
        free(TLlama_loaded_tokens);
        TLlama_loaded_tokens = NULL;
        TLlama_loaded_count = 0;

IV
llama_state_seq_load_file_token(IV idx)
    CODE:
        int32_t i = (int32_t)idx;
        if (!TLlama_loaded_tokens || i < 0 || i >= TLlama_loaded_count) {
            RETVAL = -1;
        } else {
            RETVAL = (IV)TLlama_loaded_tokens[i];
        }
    OUTPUT:
        RETVAL

# ============================================================================
# Extended per-sequence state — get size with flags
# ============================================================================

UV
llama_state_seq_get_size_ext(IV ctx, IV seq_id, IV flags)
    CODE:
        RETVAL = (UV)llama_state_seq_get_size_ext((struct llama_context*)ctx, (llama_seq_id)seq_id, (llama_state_seq_flags)flags);
    OUTPUT:
        RETVAL

# ============================================================================
# Extended per-sequence state — get data with flags
# ============================================================================

SV*
llama_state_seq_get_data_ext(IV ctx, IV seq_id, IV flags)
    CODE:
        size_t size = llama_state_seq_get_size_ext((struct llama_context*)ctx, (llama_seq_id)seq_id, (llama_state_seq_flags)flags);
        uint8_t* buf = (uint8_t*)malloc(size);
        size_t copied = llama_state_seq_get_data_ext((struct llama_context*)ctx, buf, size, (llama_seq_id)seq_id, (llama_state_seq_flags)flags);
        RETVAL = newSVpv((char*)buf, copied);
        free(buf);
    OUTPUT:
        RETVAL

# ============================================================================
# Extended per-sequence state — set data with flags
# ============================================================================

UV
llama_state_seq_set_data_ext(IV ctx, SV* data_sv, IV seq_id, IV flags)
    CODE:
        STRLEN len;
        char* data = SvPV(data_sv, len);
        RETVAL = (UV)llama_state_seq_set_data_ext((struct llama_context*)ctx, (const uint8_t*)data, len, (llama_seq_id)seq_id, (llama_state_seq_flags)flags);
    OUTPUT:
        RETVAL

# ============================================================================
# Memory handle
# ============================================================================

IV
llama_get_memory(IV ctx)
    CODE:
        RETVAL = (IV)llama_get_memory((const struct llama_context*)ctx);
    OUTPUT:
        RETVAL

# ============================================================================
# Memory — clear
# ============================================================================

void
llama_memory_clear(IV mem, bool data)
    CODE:
        llama_memory_clear((llama_memory_t)mem, data);

# ============================================================================
# Memory — seq_rm
# ============================================================================

bool
llama_memory_seq_rm(IV mem, IV seq_id, IV p0, IV p1)
    CODE:
        RETVAL = llama_memory_seq_rm((llama_memory_t)mem, (llama_seq_id)seq_id, (llama_pos)p0, (llama_pos)p1);
    OUTPUT:
        RETVAL

# ============================================================================
# Memory — seq_cp
# ============================================================================

void
llama_memory_seq_cp(IV mem, IV seq_id_src, IV seq_id_dst, IV p0, IV p1)
    CODE:
        llama_memory_seq_cp((llama_memory_t)mem, (llama_seq_id)seq_id_src, (llama_seq_id)seq_id_dst, (llama_pos)p0, (llama_pos)p1);

# ============================================================================
# Memory — seq_keep
# ============================================================================

void
llama_memory_seq_keep(IV mem, IV seq_id)
    CODE:
        llama_memory_seq_keep((llama_memory_t)mem, (llama_seq_id)seq_id);

# ============================================================================
# Memory — seq_add
# ============================================================================

void
llama_memory_seq_add(IV mem, IV seq_id, IV p0, IV p1, IV delta)
    CODE:
        llama_memory_seq_add((llama_memory_t)mem, (llama_seq_id)seq_id, (llama_pos)p0, (llama_pos)p1, (llama_pos)delta);

# ============================================================================
# Memory — seq_div
# ============================================================================

void
llama_memory_seq_div(IV mem, IV seq_id, IV p0, IV p1, IV d)
    CODE:
        llama_memory_seq_div((llama_memory_t)mem, (llama_seq_id)seq_id, (llama_pos)p0, (llama_pos)p1, (int)d);

# ============================================================================
# Memory — seq_pos_min
# ============================================================================

IV
llama_memory_seq_pos_min(IV mem, IV seq_id)
    CODE:
        RETVAL = (IV)llama_memory_seq_pos_min((llama_memory_t)mem, (llama_seq_id)seq_id);
    OUTPUT:
        RETVAL

# ============================================================================
# Memory — seq_pos_max
# ============================================================================

IV
llama_memory_seq_pos_max(IV mem, IV seq_id)
    CODE:
        RETVAL = (IV)llama_memory_seq_pos_max((llama_memory_t)mem, (llama_seq_id)seq_id);
    OUTPUT:
        RETVAL

# ============================================================================
# Memory — can_shift
# ============================================================================

bool
llama_memory_can_shift(IV mem)
    CODE:
        RETVAL = llama_memory_can_shift((llama_memory_t)mem);
    OUTPUT:
        RETVAL
