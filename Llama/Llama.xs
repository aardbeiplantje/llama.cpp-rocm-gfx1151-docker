#define PERL_NO_GET_CONTEXT

#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"
#define PERLIO_NOT_STDIO 0    /* For co-existence with stdio only */
#include <perlio.h>           /* Usually via #include <perl.h> */

#include "string.h"
#include "llama.h"

static llama_token* TLlama_tokens = NULL;
static int32_t TLlama_token_count = 0;
static STRLEN TLlama_text_len = 0;
static llama_token* TLlama_loaded_tokens = NULL;
static size_t     TLlama_loaded_count = 0;

/* Global storage for Perl log callback code reference */
static SV *log_cb_sv = NULL;   /* The coderef $cb->($level, $message, $data?) */
/* ggml_log_callback type definition */  
typedef void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data);



#define THISSvOK(sv)  (sv != NULL && SvROK(sv) && SvOK(SvRV(sv)) && INT2PTR(void *, SvIV(SvRV(sv))) != NULL)
#define THIS(sv)      INT2PTR(void *, SvIV(SvRV(sv)))
#define LLM(sv)       ((p_llama_model *)THIS(sv))->llm
typedef struct {
  llama_model *llm = NULL;
} p_llama_model;

/* No-op log callback for disabling logs when no custom handler set */
static void llama_log_noop(enum ggml_log_level level, const char * text, void * user_data) {
    (void)level; (void)text; (void)user_data;
}

static void llama_log_custom_via_perl(enum ggml_log_level level, const char * text, void * userp) {
    dTHX;
    dSP;
    if (!text || !log_cb_sv || SvTYPE(log_cb_sv) != SVt_PVCV) {
        return;  /* Not initialized or invalid cb - silently ignore */
    }

    ENTER;      /* Save old scope state before calling into Perl code */
    SAVETMPS;   /* Mark temporary values to be freed after LEAVE */

    PUSHMARK(SP);                            /* Start pushing arguments onto Perl call stack */
    mXPUSHs(newSVpv(text, strlen(text)));    /* Arg1: message string */
    mXPUSHs(newSViv((IV)level));             /* Arg2: log level enum as integer */

    PUTBACK;     /* Write back SP so Perl can see our pushed args */

    /* Call the stored perl coderef with eval wrapper to catch exceptions */  
    int count = call_sv(log_cb_sv, G_EVAL | G_DISCARD | G_KEEPERR);  /* void return expected

    SPAGAIN;     /* Refresh stack pointer after call returns */
    SV *err_tmp = ERRSV;              /* Check $@ for any errors from callback execution */
    if (SvTRUE(err_tmp)) POPs;        /* Pop error value off stack to avoid leak */

    FREETMPS;   /* Free temporaries created between ENTER/SAVETMPS and here */  
    LEAVE;      /* Restore old scope state - cleanup done by SAVETMPS/FREETMPS pair above*/
}

MODULE = Llama            PACKAGE = Llama               PREFIX = L_
VERSIONCHECK: DISABLE
PROTOTYPES: DISABLE


void
L_set_log_callback(SV *cb, ...)
    PREINIT:
        int items_arg;
    PPCODE:
        dTHX;
        dSP;

        /* Validate coderef */
        if (!cb || !SvROK(cb) || SvTYPE(SvRV(cb)) != SVt_PVCV)
            croak("callback must be a code reference");

        /* first increase refcount, then decrease the old one, else we
           might GC the object while we are just reusing the same var */
        SV *svcb = SvRV(cb);
        SvREFCNT_inc(svcb);
        if (log_cb_sv){
            SvREFCNT_dec(log_cb_sv);
            log_cb_sv = NULL;
        }
        log_cb_sv = svcb;

void
L_backend_init()
    PPCODE:
        dTHX;
        dSP;

        /* Priority order for logging configuration */
        /* 1. Custom Perl callback via set_log_callback() takes highest precedence */  
        if (log_cb_sv) {
            llama_log_set((ggml_log_callback)llama_log_custom_via_perl, NULL);
            llama_backend_init();
            return;   /* Done - no env var check needed when custom handler installed */
        }
        
        /* 2. Environment variable controls whether logs are suppressed or shown */
        const char *log_enable_env = PerlEnv_getenv("LLAMA_LOG_ENABLE");
        int should_log = 0; /* default: disabled/suppressed */
        
        if (log_enable_env && (strcmp(log_enable_env, "1") == 0 || strcmp(log_enable_env, "true") == 0)) {
            should_log = 1; /* enabled by user request like LLAMA_LOG_ENABLE=1 in shell */
        }
        
        if (should_log) {
            llama_log_set(NULL, NULL); /* use default logger outputs to stderr/stdout */
        } else {
            llama_log_set((ggml_log_callback)llama_log_noop, NULL); /* suppress all llama.cpp logging output */
        }
        llama_backend_init();

void
L_backend_free()
    CODE:
        /* Clean up stored callbacks on backend shutdown */
        if (log_cb_sv){
            SvREFCNT_dec(log_cb_sv);
            log_cb_sv = NULL;
        }
        llama_backend_free();

# ============================================================================
# Model loading / freeing
# ============================================================================

void
L_model_load(SV *file=&PL_sv_undef)
    PPCODE:
        dTHX;
        dSP;
        if(!file || !SvOK(file) || !SvPOK(file))
            XSRETURN_UNDEF;
        //printf("MODEL CREATE 0 FILE %s\n", SvPVX(file));
        struct llama_model_params params = llama_model_default_params();
        params.n_gpu_layers = 999999;
        params.use_mmap = true;
        llama_model *m = llama_model_load_from_file(SvPVX(file), params);
        if(!m)
            XSRETURN_UNDEF;
        //printf("MODEL CREATE 1 FILE %s\n", SvPVX(file));
        void *ptr = NULL;
        Newxz(ptr, 1, p_llama_model);
        //printf("MODEL CREATE 1 PTR %p\n", ptr);
        if(!ptr)
            XSRETURN_IV(0);
        SV *sv = sv_newmortal();
        SvPOK_only(sv);
        sv_setref_pv(sv, "Llama::Model", ptr);
        SvREADONLY_on(sv);
        LLM(sv) = m;
        ST(0) = sv;
        XSRETURN(1);


void
L_llama_model_free(SV *m=NULL)
    PPCODE:
        dTHX;
        dSP;
        //printf("MODEL FREE 1 %p\n", m);
        if(!THISSvOK(m))
            XSRETURN_UNDEF;
        //printf("MODEL FREE 2a %p\n", THIS(m));
        if(LLM(m) != NULL){
            llama_model_free((struct llama_model*)(LLM(m)));
            //printf("MODEL FREE 2b %p\n", THIS(m));
        }
        LLM(m) = NULL;

# ============================================================================
# Model query functions
# ============================================================================

void
L_llama_model_n_ctx_train(SV *m=NULL)
    PPCODE:
        dTHX;
        dSP;
        if(!THISSvOK(m))
            XSRETURN_UNDEF;
        XSRETURN_IV((IV)llama_model_n_ctx_train(LLM(m)));

void
L_llama_model_n_embd(SV *m=NULL)
    PPCODE:
        dTHX;
        dSP;
        if(!THISSvOK(m))
            XSRETURN_UNDEF;
        XSRETURN_IV((IV)llama_model_n_embd(LLM(m)));

void
L_llama_model_n_layer(SV *m=NULL)
    PPCODE:
        dTHX;
        dSP;
        if(!THISSvOK(m))
            XSRETURN_UNDEF;
        XSRETURN_IV((IV)llama_model_n_layer(LLM(m)));

void
L_llama_model_n_params(SV *m=NULL)
    PPCODE:
        dTHX;
        dSP;
        if(!THISSvOK(m))
            XSRETURN_UNDEF;
        XSRETURN_IV((UV)llama_model_n_params(LLM(m)));

void
L_llama_model_size(SV *m=NULL)
    PPCODE:
        dTHX;
        dSP;
        if(!THISSvOK(m))
            XSRETURN_UNDEF;
        XSRETURN_IV((UV)llama_model_size(LLM(m)));

IV
L_llama_model_desc(SV *m=NULL)
    PPCODE:
        dTHX;
        dSP;
        if(!THISSvOK(m))
            XSRETURN_UNDEF;
        char *buf = NULL;
        Newxz(buf, 4096, char);
        if(!buf)
            XSRETURN_UNDEF;
        llama_model_desc(LLM(m), (char *)buf, 4096);

        HV *rh = newHV();

        // Parse key-value pairs separated by \0 bytes
        char *ptr = buf;
        char *end = buf + 4096;

        while (ptr < end && *ptr != '\0') {
            char *key = ptr;
            STRLEN key_len = strlen(key);
            ptr += key_len + 1;

            if (ptr >= end) break; // Safeguard against malformed/truncated buffer

            char *val = ptr;
            STRLEN val_len = strlen(val);
            ptr += val_len + 1;

            // Store key-value pair into the Perl Hash
            hv_store(rh, key, key_len, newSVpvn(val, val_len), 0);
        }

        // Free allocated memory buffer
        Safefree(buf);

        // Return the hash reference (use newRV_noinc to avoid leaking the HV reference count)
        ST(0) = newRV_noinc((SV *)rh);
        sv_2mortal(ST(0));
        XSRETURN(1);

const char*
L_llama_model_chat_template(IV model, char* name)
    CODE:
        RETVAL = llama_model_chat_template((const struct llama_model*)model, name);
    OUTPUT:
        RETVAL

SV*
L_llama_chat_apply_template(SV* tmpl_sv, SV* messages_sv, bool add_ass)
    PREINIT:
        const char *tmpl;
        AV *messages_av;
        int32_t n_msg;
        struct llama_chat_message *chat_arr;
        STRLEN len;
        char *buf;
        int32_t ret_len;
        SV *result;
    PPCODE:
        /* Get template string */
        if (SvOK(tmpl_sv)) {
            tmpl = SvPV(tmpl_sv, len);
        } else {
            tmpl = NULL;  /* Use built-in default */
        }

        /* Parse messages array - expect ARRAYREF of HASHREFs with role/content keys */
        if (!SvROK(messages_sv) || SvTYPE(SvRV(messages_sv)) != SVt_PVAV) {
            croak("messages must be an array reference");
        }
        messages_av = (AV*)SvRV(messages_sv);
        n_msg = av_len(messages_av) + 1;

        /* Allocate C array for chat messages */
        chat_arr = (struct llama_chat_message *)malloc(n_msg * sizeof(struct llama_chat_message));

        /* Extract role and content from each message hashref */
        for (int i = 0; i < n_msg; i++) {
            SV **msg_ref = av_fetch(messages_av, i, 0);
            if (!msg_ref || !SvROK(*msg_ref)) {
                free(chat_arr);
                croak("message %d is not a hash reference", i);
            }

            HV *msg_hv = (HV*)SvRV(*msg_ref);
            SV **role_sv = hv_fetch(msg_hv, "role", 4, 0);
            SV **content_sv = hv_fetch(msg_hv, "content", 7, 0);

            if (!role_sv || !*role_sv || !content_sv || !*content_sv) {
                free(chat_arr);
                croak("message %d missing 'role' or 'content'", i);
            }

            chat_arr[i].role    = SvPV_nolen(*role_sv);
            chat_arr[i].content = SvPV_nolen(*content_sv);
        }

        /* First call: get required buffer size */
        ret_len = llama_chat_apply_template(tmpl, chat_arr, n_msg, add_ass, NULL, 0);
        if (ret_len <= 0) {
            free(chat_arr);
            XSRETURN_EMPTY;
        }

        /* Allocate output string with extra space for null terminator */
        result = newSV(ret_len + 1);
        buf = SvPVX(result);

        /* Second call: write formatted prompt to buffer */
        ret_len = llama_chat_apply_template(tmpl, chat_arr, n_msg, add_ass, buf, ret_len + 1);

        /* Clean up and return result */
        free(chat_arr);
        SvCUR_set(result, ret_len);
        *SvEND(result) = '\0';

        ST(0) = sv_2mortal(result);
        XSRETURN(1);

# ============================================================================
# Context creation / destruction
# ============================================================================

void
L_llama_init_from_model(SV *m=NULL, IV n_ctx, IV n_batch, IV n_threads, IV n_threads_batch, bool embeddings)
    PPCODE:
        dTHX;
        dSP;
        if(!THISSvOK(m))
            XSRETURN_UNDEF;
        struct llama_context_params params = llama_context_default_params();
        params.n_ctx = (uint32_t)n_ctx;
        params.n_batch = (uint32_t)n_batch;
        params.n_ubatch = 256;
        params.n_seq_max = 1;
        params.n_threads = (int32_t)n_threads;
        params.n_threads_batch = (int32_t)n_threads_batch;
        params.embeddings = embeddings;
        XSRETURN_IV((IV)llama_init_from_model(LLM(m), params));

void
L_llama_free(IV ctx)
    CODE:
        llama_free((struct llama_context*)ctx);

# ============================================================================
# Context query
# ============================================================================

IV
L_llama_n_ctx(IV ctx)
    CODE:
        RETVAL = (IV)llama_n_ctx((const struct llama_context*)ctx);
    OUTPUT:
        RETVAL

IV
L_llama_n_batch(IV ctx)
    CODE:
        RETVAL = (IV)llama_n_batch((const struct llama_context*)ctx);
    OUTPUT:
        RETVAL

IV
L_llama_n_seq_max(IV ctx)
    CODE:
        RETVAL = (IV)llama_n_seq_max((const struct llama_context*)ctx);
    OUTPUT:
        RETVAL

# ============================================================================
# Batch management
# ============================================================================

IV
L_llama_batch_init(IV n_tokens, IV embd, IV n_seq_max)
    CODE:
        struct llama_batch batch = llama_batch_init((int32_t)n_tokens, (int32_t)embd, (int32_t)n_seq_max);
        RETVAL = (IV)malloc(sizeof(struct llama_batch));
        *(struct llama_batch*)RETVAL = batch;
    OUTPUT:
        RETVAL

void
L_llama_batch_free(IV batch)
    CODE:
        llama_batch_free(*(struct llama_batch*)batch);
        free((void*)batch);

# ============================================================================
# Batch field setters
# ============================================================================

void
L_llama_batch_set_n_tokens(IV batch, IV n_tokens)
    CODE:
        struct llama_batch* b = (struct llama_batch*)batch;
        b->n_tokens = (int32_t)n_tokens;

void
L_llama_batch_set_token(IV batch, IV idx, IV token, IV pos, SV* seq_id_sv)
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
L_llama_decode(IV ctx, IV batch)
    CODE:
        RETVAL = (IV)llama_decode((struct llama_context*)ctx, *(struct llama_batch*)batch);
    OUTPUT:
        RETVAL

IV
L_llama_encode(IV ctx, IV batch)
    CODE:
        RETVAL = (IV)llama_encode((struct llama_context*)ctx, *(struct llama_batch*)batch);
    OUTPUT:
        RETVAL

# ============================================================================
# Logits retrieval — returns pointer to float array
# ============================================================================

float*
L_llama_get_logits(IV ctx)
    CODE:
        RETVAL = llama_get_logits((struct llama_context*)ctx);
    OUTPUT:
        RETVAL

# ============================================================================
# Logits retrieval (ith token) — returns single float
# ============================================================================

NV
L_llama_get_logits_ith(IV ctx, IV i)
    CODE:
        float* logits = llama_get_logits_ith((struct llama_context*)ctx, (int32_t)i);
        RETVAL = logits ? (NV)logits[0] : 0.0;
    OUTPUT:
        RETVAL

# ============================================================================
# Embeddings retrieval — returns pointer to float array
# ============================================================================

float*
L_llama_get_embeddings(IV ctx)
    CODE:
        RETVAL = llama_get_embeddings((struct llama_context*)ctx);
    OUTPUT:
        RETVAL

# ============================================================================
# Tokenization — persistent buffer for tokenized results
# ============================================================================

IV
L_llama_tokenize(IV vocab, SV* text, IV max_tokens, bool add_special, bool parse_special)
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
L_llama_tokenize_get_token(IV idx)
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
L_llama_tokenize_free()
    CODE:
        free(TLlama_tokens);
        TLlama_tokens = NULL;
        TLlama_token_count = 0;

# ============================================================================
# Token to text piece
# ============================================================================

IV
L_llama_token_to_piece(IV vocab, IV token, char* buf, IV buf_len, IV lstrip, bool special)
    CODE:
        RETVAL = (IV)llama_token_to_piece((const struct llama_vocab*)vocab, (llama_token)token, buf, (int32_t)buf_len, (int32_t)lstrip, special);
    OUTPUT:
        RETVAL

# ============================================================================
# Detokenization (tokens → text)
# ============================================================================

IV
L_llama_detokenize(IV vocab, IV* tokens, IV n_tokens, char* buf, IV buf_len, bool remove_special, bool unparse_special)
    CODE:
        RETVAL = (IV)llama_detokenize((const struct llama_vocab*)vocab, (const llama_token*)tokens, (int32_t)n_tokens, buf, (int32_t)buf_len, remove_special, unparse_special);
    OUTPUT:
        RETVAL

# ============================================================================
# Vocabulary query
# ============================================================================

IV
L_llama_vocab_n_tokens(IV vocab)
    CODE:
        RETVAL = (IV)llama_vocab_n_tokens((const struct llama_vocab*)vocab);
    OUTPUT:
        RETVAL

llama_token
L_llama_vocab_bos(IV vocab)
    CODE:
        RETVAL = llama_vocab_bos((const struct llama_vocab*)vocab);
    OUTPUT:
        RETVAL

llama_token
L_llama_vocab_eos(IV vocab)
    CODE:
        RETVAL = llama_vocab_eos((const struct llama_vocab*)vocab);
    OUTPUT:
        RETVAL

llama_token
L_llama_vocab_eot(IV vocab)
    CODE:
        RETVAL = llama_vocab_eot((const struct llama_vocab*)vocab);
    OUTPUT:
        RETVAL

llama_token
L_llama_vocab_nl(IV vocab)
    CODE:
        RETVAL = llama_vocab_nl((const struct llama_vocab*)vocab);
    OUTPUT:
        RETVAL

# ============================================================================
# Sampling — individual samplers
# ============================================================================

IV
L_greedy_sampler()
    CODE:
        RETVAL = (IV)llama_sampler_init_greedy();
    OUTPUT:
        RETVAL

IV
L_sampler_init_top_k(IV k)
    CODE:
        RETVAL = (IV)llama_sampler_init_top_k((int32_t)k);
    OUTPUT:
        RETVAL

IV
L_sampler_init_top_p(NV p, IV min_keep)
    CODE:
        RETVAL = (IV)llama_sampler_init_top_p((float)p, (size_t)min_keep);
    OUTPUT:
        RETVAL

IV
L_sampler_init_temp(NV t)
    CODE:
        RETVAL = (IV)llama_sampler_init_temp((float)t);
    OUTPUT:
        RETVAL

IV
L_dist_sampler(IV seed)
    CODE:
        RETVAL = (IV)llama_sampler_init_dist((uint32_t)seed);
    OUTPUT:
        RETVAL

# ============================================================================
# Sampler chain
# ============================================================================

IV
L_llama_sampler_chain_init()
    CODE:
        struct llama_sampler_chain_params params = llama_sampler_chain_default_params();
        RETVAL = (IV)llama_sampler_chain_init(params);
    OUTPUT:
        RETVAL

void
L_llama_sampler_chain_add(IV chain, IV sampler)
    CODE:
        llama_sampler_chain_add((struct llama_sampler*)chain, (struct llama_sampler*)sampler);

IV
L_llama_sampler_chain_get(IV chain, IV i)
    CODE:
        RETVAL = (IV)llama_sampler_chain_get((struct llama_sampler*)chain, (int32_t)i);
    OUTPUT:
        RETVAL

IV
L_llama_sampler_chain_n(IV chain)
    CODE:
        RETVAL = (IV)llama_sampler_chain_n((const struct llama_sampler*)chain);
    OUTPUT:
        RETVAL

void
L_llama_sampler_free(IV sampler)
    CODE:
        llama_sampler_free((struct llama_sampler*)sampler);

void
L_llama_sampler_chain_free(IV chain)
    CODE:
        llama_sampler_free((struct llama_sampler*)chain);

# ============================================================================
# Sampler apply — sample a token from context
# ============================================================================

llama_token
L_llama_sampler_sample(IV sampler, IV ctx, IV idx)
    CODE:
        RETVAL = llama_sampler_sample((struct llama_sampler*)sampler, (struct llama_context*)ctx, (int32_t)idx);
    OUTPUT:
        RETVAL

void
L_llama_sampler_reset(IV sampler)
    CODE:
        llama_sampler_reset((struct llama_sampler*)sampler);

# ============================================================================
# Performance info
# ============================================================================

NV
L_llama_perf_context_load_ms(IV ctx)
    CODE:
        struct llama_perf_context_data data = llama_perf_context((const struct llama_context*)ctx);
        RETVAL = data.t_load_ms;
    OUTPUT:
        RETVAL

NV
L_llama_perf_context_eval_ms(IV ctx)
    CODE:
        struct llama_perf_context_data data = llama_perf_context((const struct llama_context*)ctx);
        RETVAL = data.t_eval_ms;
    OUTPUT:
        RETVAL

void
L_llama_perf_context_reset(IV ctx)
    CODE:
        llama_perf_context_reset((struct llama_context*)ctx);

# ============================================================================
# Model → vocab query
# ============================================================================

void
L_llama_model_get_vocab(SV *m=NULL)
    PPCODE:
        dTHX;
        dSP;
        if(!THISSvOK(m))
            XSRETURN_UNDEF;
        XSRETURN_IV((IV)llama_model_get_vocab(LLM(m)));

# ============================================================================
# Context → model query
# ============================================================================

IV
L_llama_get_model(IV ctx)
    CODE:
        RETVAL = (IV)(IV)llama_get_model((const struct llama_context*)ctx);
    OUTPUT:
        RETVAL

# ============================================================================
# KV cache state — get size
# ============================================================================

UV
L_llama_state_get_size(IV ctx)
    CODE:
        RETVAL = (UV)llama_state_get_size((struct llama_context*)ctx);
    OUTPUT:
        RETVAL

# ============================================================================
# KV cache state — get data into packed binary string
# ============================================================================

SV*
L_llama_state_get_data(IV ctx)
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
L_llama_state_set_data(IV ctx, SV* data_sv)
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
L_llama_state_save_file(IV ctx, SV* path_sv, SV* tokens_sv)
    CODE:
        STRLEN path_len;
        char* path = SvPV(path_sv, path_len);
        AV* tokens_av = NULL;
        if (SvROK(tokens_sv) && SvTYPE(SvRV(tokens_sv)) == SVt_PVAV) {
            tokens_av = (AV*)SvRV(tokens_sv);
        }
        size_t n = tokens_av ? (av_len(tokens_av) + 1) : 0;
        llama_token* tok_buf = (llama_token*)malloc(n * sizeof(llama_token));
        if (tokens_av) {
            for (size_t i = 0; i < n; i++) {
                SV* sv = *av_fetch(tokens_av, (int)i, 0);
                tok_buf[i] = (llama_token)SvIV(sv);
            }
        }
        RETVAL = llama_state_save_file((struct llama_context*)ctx, path, tok_buf, n);
        free(tok_buf);
    OUTPUT:
        RETVAL

# ============================================================================
# KV cache state — load session file
# ============================================================================

bool
L_llama_state_load_file(IV ctx, SV* path_sv, IV capacity)
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
L_llama_state_load_file_count()
    CODE:
        RETVAL = (IV)TLlama_loaded_count;
    OUTPUT:
        RETVAL

void
L_llama_state_load_file_free()
    CODE:
        free(TLlama_loaded_tokens);
        TLlama_loaded_tokens = NULL;
        TLlama_loaded_count = 0;

IV
L_llama_state_load_file_token(IV idx)
    CODE:
        int32_t i = (int32_t)idx;
        if (!TLlama_loaded_tokens || i < 0 || (size_t)i >= TLlama_loaded_count) {
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
L_llama_state_seq_get_size(IV ctx, IV seq_id)
    CODE:
        RETVAL = (UV)llama_state_seq_get_size((struct llama_context*)ctx, (llama_seq_id)seq_id);
    OUTPUT:
        RETVAL

# ============================================================================
# Per-sequence state — get data
# ============================================================================

SV*
L_llama_state_seq_get_data(IV ctx, IV seq_id)
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
L_llama_state_seq_set_data(IV ctx, SV* data_sv, IV seq_id)
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
L_llama_state_seq_save_file(IV ctx, SV* path_sv, IV seq_id, SV* tokens_sv)
    CODE:
        STRLEN path_len;
        char* path = SvPV(path_sv, path_len);
        AV* tokens_av = NULL;
        if (SvROK(tokens_sv) && SvTYPE(SvRV(tokens_sv)) == SVt_PVAV) {
            tokens_av = (AV*)SvRV(tokens_sv);
        }
        size_t n = tokens_av ? (av_len(tokens_av) + 1) : 0;
        llama_token* tok_buf = (llama_token*)malloc(n * sizeof(llama_token));
        if (tokens_av) {
            for (size_t i = 0; i < n; i++) {
                SV* sv = *av_fetch(tokens_av, (int)i, 0);
                tok_buf[i] = (llama_token)SvIV(sv);
            }
        }
        RETVAL = (UV)llama_state_seq_save_file((struct llama_context*)ctx, path, (llama_seq_id)seq_id, tok_buf, n);
        free(tok_buf);
    OUTPUT:
        RETVAL

# ============================================================================
# Per-sequence state — load session file
# ============================================================================

bool
L_llama_state_seq_load_file(IV ctx, SV* path_sv, IV seq_id, IV capacity)
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
L_llama_state_seq_load_file_count()
    CODE:
        RETVAL = (IV)TLlama_loaded_count;
    OUTPUT:
        RETVAL

void
L_llama_state_seq_load_file_free()
    CODE:
        free(TLlama_loaded_tokens);
        TLlama_loaded_tokens = NULL;
        TLlama_loaded_count = 0;

IV
L_llama_state_seq_load_file_token(IV idx)
    CODE:
        int32_t i = (int32_t)idx;
        if (!TLlama_loaded_tokens || i < 0 || (size_t)i >= TLlama_loaded_count) {
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
L_llama_state_seq_get_size_ext(IV ctx, IV seq_id, IV flags)
    CODE:
        RETVAL = (UV)llama_state_seq_get_size_ext((struct llama_context*)ctx, (llama_seq_id)seq_id, (llama_state_seq_flags)flags);
    OUTPUT:
        RETVAL

# ============================================================================
# Extended per-sequence state — get data with flags
# ============================================================================

SV*
L_llama_state_seq_get_data_ext(IV ctx, IV seq_id, IV flags)
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
L_llama_state_seq_set_data_ext(IV ctx, SV* data_sv, IV seq_id, IV flags)
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
L_llama_get_memory(IV ctx)
    CODE:
        RETVAL = (IV)llama_get_memory((const struct llama_context*)ctx);
    OUTPUT:
        RETVAL

# ============================================================================
# Memory — clear
# ============================================================================

void
L_llama_memory_clear(IV mem, bool data)
    CODE:
        llama_memory_clear((llama_memory_t)mem, data);

# ============================================================================
# Memory — seq_rm
# ============================================================================

bool
L_llama_memory_seq_rm(IV mem, IV seq_id, IV p0, IV p1)
    CODE:
        RETVAL = llama_memory_seq_rm((llama_memory_t)mem, (llama_seq_id)seq_id, (llama_pos)p0, (llama_pos)p1);
    OUTPUT:
        RETVAL

# ============================================================================
# Memory — seq_cp
# ============================================================================

void
L_llama_memory_seq_cp(IV mem, IV seq_id_src, IV seq_id_dst, IV p0, IV p1)
    CODE:
        llama_memory_seq_cp((llama_memory_t)mem, (llama_seq_id)seq_id_src, (llama_seq_id)seq_id_dst, (llama_pos)p0, (llama_pos)p1);

# ============================================================================
# Memory — seq_keep
# ============================================================================

void
L_llama_memory_seq_keep(IV mem, IV seq_id)
    CODE:
        llama_memory_seq_keep((llama_memory_t)mem, (llama_seq_id)seq_id);

# ============================================================================
# Memory — seq_add
# ============================================================================

void
L_llama_memory_seq_add(IV mem, IV seq_id, IV p0, IV p1, IV delta)
    CODE:
        llama_memory_seq_add((llama_memory_t)mem, (llama_seq_id)seq_id, (llama_pos)p0, (llama_pos)p1, (llama_pos)delta);

# ============================================================================
# Memory — seq_div
# ============================================================================

void
L_llama_memory_seq_div(IV mem, IV seq_id, IV p0, IV p1, IV d)
    CODE:
        llama_memory_seq_div((llama_memory_t)mem, (llama_seq_id)seq_id, (llama_pos)p0, (llama_pos)p1, (int)d);

# ============================================================================
# Memory — seq_pos_min
# ============================================================================

IV
L_llama_memory_seq_pos_min(IV mem, IV seq_id)
    CODE:
        RETVAL = (IV)llama_memory_seq_pos_min((llama_memory_t)mem, (llama_seq_id)seq_id);
    OUTPUT:
        RETVAL

# ============================================================================
# Memory — seq_pos_max
# ============================================================================

IV
L_llama_memory_seq_pos_max(IV mem, IV seq_id)
    CODE:
        RETVAL = (IV)llama_memory_seq_pos_max((llama_memory_t)mem, (llama_seq_id)seq_id);
    OUTPUT:
        RETVAL

# ============================================================================
# Memory — can_shift
# ============================================================================

bool
L_llama_memory_can_shift(IV mem)
    CODE:
        RETVAL = llama_memory_can_shift((llama_memory_t)mem);
    OUTPUT:
        RETVAL
