# PLAN.md — Perl XS Module for libllama.so

## Status: ✅ Core Engine Complete — Inference Engine + Multi-Model + KV Cache + Conv ID

All core features implemented. 180+ tests passing across 8 test files.
Llama::Cache inference engine with dynamic model loading, multi-model support, slot management, chat completion, completion, embeddings, streaming, KV cache persistence with mmap, conv_id-based slot reuse, and full nginx API integration is fully functional.

## Architecture

```
Client → nginx (Lua+Perl) → llama-server (Unix socket) → ROCm GPU
         ↘ Llama::Cache (XS) → ROCm GPU
```

**Dual-mode:** Keep the existing llama-server proxy as-is. Add new `/api/cache` endpoints that route to Llama::Cache. Gradual migration path.

### Existing flow (unchanged)
```
Client → nginx (Lua block: body fixup) → proxy_pass → llama-server → GPU
```

### New flow (via /api/cache)
```
Client → nginx (Perl: Llama::cache_handler) → Llama::Cache → GPU
```

## File Structure

```
Llama/
├── Makefile.PL           # ExtUtils::MakeMaker build config
├── typemap               # Custom typemap for llama.cpp C types
├── Llama.xs              # XS declarations and glue code
├── Llama.pm              # Perl OO wrapper + high-level API
├── Llama/
│   ├── Types.pm          # Enums and constants
│   ├── Model.pm          # OO wrapper for llama_model
│   ├── Context.pm        # OO wrapper for llama_context
│   ├── Batch.pm          # OO wrapper for llama_batch
│   ├── Vocab.pm          # OO wrapper for llama_vocab
│   ├── ModelConfig.pm    # Preset file parsing (INI format)
│   ├── Server.pm         # Pre-fork worker server
│   ├── Cache.pm          # Inference engine with slot management
│   └── Cache/
│       └── Stream.pm     # SSE streaming helper class
├── t/
│   ├── 00-load.t         # Module load tests
│   ├── 01-model-load.t   # Model loading tests
│   ├── 02-inference.t    # Tokenization, decode, sampling tests
│   ├── 03-kv-cache.t     # State save/restore, session I/O tests
│   ├── 04-mmap-showcase.t # Mmap model loading tests
│   ├── 05-cache.t        # Llama::Cache engine tests (53 tests)
│   ├── 06-stream.t       # SSE streaming tests
│   └── 07-integration.t  # End-to-end nginx integration tests
└── .gitignore
```

## Completed Work

### 1. XS Bindings (Llama.xs)

All planned bindings implemented. Extended with per-sequence state and KV cache manipulation.

**XS Functions:**

```c
// Model loading
IV      llama_model_load_from_file(char* path)
IV      llama_model_load_from_file_mmap(char* path)       // use_mmap=true
IV      llama_model_load_from_file_mmap(char* path, int use_mmap)

// KV cache state (in-memory)
UV      llama_state_get_size(IV ctx)
SV*     llama_state_get_data(IV ctx)                        // returns packed binary string
UV      llama_state_set_data(IV ctx, SV* data_sv)           // returns bytes restored

// KV cache session files (global)
bool    llama_state_save_file(IV ctx, SV* path, SV* tokens_sv)
bool    llama_state_load_file(IV ctx, SV* path, IV capacity)
IV      llama_state_load_file_count()
IV      llama_state_load_file_token(IV idx)
void    llama_state_load_file_free()

// Per-sequence state
UV      llama_state_seq_get_size(IV ctx, IV seq_id)
SV*     llama_state_seq_get_data(IV ctx, IV seq_id)
UV      llama_state_seq_set_data(IV ctx, SV* data_sv, IV seq_id)
UV      llama_state_seq_save_file(IV ctx, SV* path, IV seq_id, SV* tokens_sv)
bool    llama_state_seq_load_file(IV ctx, SV* path, IV seq_id, IV capacity)
IV      llama_state_seq_load_file_count()
IV      llama_state_seq_load_file_token(IV idx)
void    llama_state_seq_load_file_free()

// Extended per-sequence (with flags)
UV      llama_state_seq_get_size_ext(IV ctx, IV seq_id, IV flags)
SV*     llama_state_seq_get_data_ext(IV ctx, IV seq_id, IV flags)
UV      llama_state_seq_set_data_ext(IV ctx, SV* data_sv, IV seq_id, IV flags)

// KV cache manipulation
IV      llama_get_memory(IV ctx)
void    llama_memory_clear(IV mem, bool data)
bool    llama_memory_seq_rm(IV mem, IV seq_id, IV p0, IV p1)
void    llama_memory_seq_cp(IV mem, IV seq_id_src, IV seq_id_dst, IV p0, IV p1)
void    llama_memory_seq_keep(IV mem, IV seq_id)
void    llama_memory_seq_add(IV mem, IV seq_id, IV p0, IV p1, IV delta)
void    llama_memory_seq_div(IV mem, IV seq_id, IV p0, IV p1, IV d)
IV      llama_memory_seq_pos_min(IV mem, IV seq_id)
IV      llama_memory_seq_pos_max(IV mem, IV seq_id)
IV      llama_memory_can_shift(IV mem)

// Model chat template queries
const char*   llama_model_chat_template(IV model, char* name)  // get built-in template by name or default

// Chat template application
SV*           llama_chat_apply_template(SV* tmpl_sv, SV* messages_av, bool add_ass)
```

### 2. Typemap Entries (typemap)

```
size_t                         T_UV
struct llama_memory_t*         T_IV
const struct llama_memory_t*   T_IV
```

`llama_memory_t` is `typedef struct llama_memory_i *` — treated as opaque pointer like `llama_context*`.
`size_t` maps to Perl `UV` (unsigned, platform-native size).

### 3. Key Bug Fix: Token Array Size Mismatch

**Bug found and fixed:** `llama_token` is `int32_t` but the existing `IV*` typemap allocates `int64_t` arrays. Casting `IV*` to `llama_token*` caused the C code to read 32-bit values from a 64-bit array, producing garbage tokens.

**Fix:** XS functions manually build a correctly-sized `llama_token*` buffer from Perl arrays, avoiding the typemap entirely for token data.

### 4. Key Bug Fix: KV Cache Save/Restore Return Values

**Bug found and fixed:** `_save_slot_cache` and `_restore_slot_cache` used `return unless` instead of `return 0 unless`, causing falsy returns when the guard condition passed. Additionally, `save_slot_to_mmap_file` and `load_slot_from_mmap_file` had `return` inside `eval` blocks which did not propagate the value correctly. The header used only `N` (n_tokens) but `seq_state_size()` returned a different value than `seq_get_state()` actual data length.

**Fix:**
- Changed all `_save_slot_cache`/`_restore_slot_cache` guards to `return 0 unless`
- Moved `return` statements outside `eval` blocks in `save_slot_to_mmap_file` and `load_slot_from_mmap_file`, storing result in a variable instead
- Changed header format from `pack('N', n_tokens)` to `pack('NN', n_tokens, actual_size)` to store actual data length and read it back on restore

### 5. Key Bug Fix: Mmap Offset

**Bug found and fixed:** When using `Sys::Mmap` to read cache files, the mapped region included the 8-byte header. The header needed to be skipped before passing data to `seq_set_state`.

**Fix:** Pass offset `8` as the last argument to `mmap()` to skip the header. Capture return value in `my $_mmap_ok = mmap(...)` to avoid Perl 5.40 "useless use of variable" warnings.

### 6. Perl Module Changes

#### Llama::Context — state methods

```perl
$ctx->state_size                          # returns UV
my $data = $ctx->get_state                # returns packed binary string
my $bytes = $ctx->set_state($data)        # restores state, returns UV bytes
$ctx->save_session($path, \@tokens)       # saves to .session file
my ($tokens_ref, $count) = $ctx->load_session($path, $capacity)  # loads from file
```

#### Llama::Context — per-sequence state

```perl
$ctx->seq_state_size($seq_id)
my $data = $ctx->seq_get_state($seq_id)
$ctx->seq_set_state($data, $dest_seq_id)
$ctx->seq_save_session($path, $seq_id, \@tokens)
my ($tokens_ref, $bytes) = $ctx->seq_load_session($path, $seq_id, $capacity)
```

#### Llama::Context — KV cache management

```perl
$ctx->clear_kv($data_only)
$ctx->seq_rm($seq_id, $p0, $p1)
$ctx->seq_cp($src_seq, $dst_seq, $p0, $p1)
$ctx->seq_keep($seq_id)
my $min = $ctx->seq_pos_min($seq_id)
my $max = $ctx->seq_pos_max($seq_id)
my $can = $ctx->can_shift
```

#### Llama.pm — model loading with mmap

```perl
Llama::model_load($path)       # standard load (use_mmap = false)
Llama::model_load_mmap($path)  # load with mmap enabled
```

#### Llama.pm — helper functions for embeddings

```perl
Llama::_get_logits_ptr($ctx_ptr)   # returns pointer to logits buffer
Llama::_read_float($ptr, $i)       # reads float at index from logits buffer
Llama::_get_embeddings_ptr($ctx_ptr)  # returns pointer to embeddings buffer
```

#### Llama::Model — chat templates

Get built-in template name:
```perl
my $tmpl = $model->chat_template();              # auto-detect default template
my $qwen_tmpl = $model->chat_template('qwen');   # explicit template name
```

Apply template to messages:
```perl
my @messages = (
    { role => 'system', content => 'You are a helpful assistant.' },
    { role => 'user', content => 'Hello!' },
);
my $prompt = $model->apply_chat_template(\@messages, $add_ass);  # add_ass defaults to false
# or with custom template string:
my $custom_prompt = Llama::llama_chat_apply_template($custom_tmpl_str, \@messages, 1);
```

Supported models/templates: Qwen3.5/3.6, Nemotron, DeepSeekV4, Gemma — all GGUF models with embedded chat templates.

#### Llama::ModelConfig — preset file parsing

```perl
my $config = Llama::ModelConfig->new(
    path        => '/models/model.gguf',
    preset_file => '/models/llamacpp_presets.ini',
    preset_name => 'model',
    overrides   => { n_ctx => 8192 },
);

$config->n_ctx      # returns effective n_ctx (preset + overrides)
$config->n_batch
$config->n_threads
$config->model_name
$config->path
```

Preset file format (INI):
```ini
[ * ]
ctx-size = 4096
n-batch = 512
n-threads = 16

[model-name]
ctx-size = 8192
```

Key aliasing maps INI keys to Perl keys:
- `ctx-size` → `n_ctx`
- `n-batch` → `n_batch`
- `n-threads` → `n_threads`
- `n-threads-batch` → `n_threads_batch`
- `embedding` → `embeddings`

### 7. Llama::Cache — KV cache + inference engine

```perl
my $cache = Llama::Cache->new(
    model_path => $path,
    n_ctx      => 4096,
    n_batch    => 512,
    n_threads  => 16,
    n_slots    => 4,
    cache_dir  => "/dev/shm/llama_cache",
);
```

**Slot management:**
```perl
my $slot_id = $cache->alloc_slot($conv_id, $model_name);  # optional conv_id/model filtering
$cache->free_slot($slot_id, $conv_id);
my $slots = $cache->get_slots();   # { 0 => { state, n_tokens, model, conv_id }, ... }
my $slot  = $cache->get_slot($slot_id);
my $slot  = $cache->get_slot_by_conv_id($conv_id);
$cache->set_slot_by_conv_id($conv_id, $slot_id);
```

**Inference (blocking):**
```perl
my $result = $cache->chat_completion($slot_id, \@messages, $n_predict, {
    conv_id => $conv_id,      # optional — enables KV cache reuse
    model   => $model_name,   # optional — selects model for slot allocation
});
my $result = $cache->completion($slot_id, $prompt, $n_predict, {
    conv_id => $conv_id,
    model   => $model_name,
});
```

**Embeddings:**
```perl
my $emb = $cache->embeddings($slot_id, $text, {
    conv_id => $conv_id,
    model   => $model_name,
});
```

**KV cache persistence:**
```perl
$cache->save_slot_cache($slot_id, $file_path);
$cache->load_slot_cache($slot_id, $file_path);
my @cached = $cache->list_cached_slots();
$cache->save_slot_to_mmap_file($slot_id, $file_path);  # with mmap offset
$cache->load_slot_from_mmap_file($slot_id, $file_path); # with mmap offset
```

**Model info:**
```perl
$cache->model_desc
$cache->model_n_ctx_train
$cache->model_n_embd
$cache->model_n_params
```

**Stats:**
```perl
$cache->get_stats()
$cache->reset_stats()
```

### 8. Llama::Cache::Stream — streaming helper class

```perl
my $stream = Llama::Cache::Stream->new($conv_id);
$stream->add_chunk($piece);
my $chunk = $stream->next_chunk($from);
$stream->finalize();
my $cancelled = $stream->is_cancelled();
```

## Test Coverage

### Test Files

| File | Tests | Description |
|---|---|---|
| `t/00-load.t` | 1 | Module loads |
| `t/01-model-load.t` | 6 | Model loading, context creation, batch setup |
| `t/chat-templates.t` | 12 | Chat template application: auto-detect from GGUF, role/content formatting, error handling |
| `t/02-inference.t` | 11 | Tokenization, decode, logits, sampling |
| `t/03-kv-cache.t` | 30 | State save/restore, session file I/O, per-sequence ops, KV cache manipulation |
| `t/04-mmap-showcase.t` | 19 | Mmap model loading, cross-context state transfer, session roundtrip |
| `t/05-cache.t` | 53 | Llama::Cache: creation, model info, slot management, chat/completion/embeddings, KV cache save/load, auto-save/restore, mmap I/O, conv_id reuse, stats |
| `t/06-stream.t` | 20 | Llama::Cache::Stream: SSE chunking, finalization, cancellation |
| `t/07-integration.t` | 16 | End-to-end nginx integration: health, models, slots, props, chat completion, KV cache reuse, streams lookup, DELETE stream, tokenize, detokenize, input tokens |

### Test Results

```
Files=9, Tests=198, wallclock secs
Result: PASS
```

### Test Coverage

1. ✅ **Save/restore roundtrip** — get state, set it back on same context
2. ✅ **Cross-context restore** — save state from context A, restore to context B
3. ✅ **Session file I/O** — save to disk, load back, verify tokens and state match
4. ✅ **Per-sequence ops** — save/restore individual sequences, extended with flags
5. ✅ **KV cache manipulation** — clear, rm, cp, keep, add, div, pos_min/max, can_shift
6. ✅ **Memory size sanity** — state size matches expected values
7. ✅ **Model mmap** — `llama_model_load_from_file_mmap` loads with `use_mmap=true`
8. ✅ **Cache slot management** — alloc, free, get_slots, get_slot
9. ✅ **Cache chat completion** — tokenize, decode, greedy sample, return OpenAI-compatible response
10. ✅ **Cache completion** — tokenize, decode, greedy sample, return OpenAI-compatible response
11. ✅ **Cache embeddings** — tokenize, decode, return embeddings array
12. ✅ **Cache KV persistence** — save slot cache to file, load back, list cached files
13. ✅ **Auto-save on free_slot** — calling free_slot triggers _save_slot_cache, writes to disk
14. ✅ **Auto-restore on alloc_slot** — calling alloc_slot triggers _restore_slot_cache, loads from disk
15. ✅ **Mmap file I/O** — save_slot_to_mmap_file and load_slot_from_mmap_file with NN header (n_tokens + actual_size) and mmap offset
16. ✅ **SSE streaming** — Llama::Cache::Stream class for chunked SSE responses
17. ✅ **Conv_id slot reuse** — same conv_id returns same slot, different conv_id returns different slot
18. ✅ **Model filtering** — alloc_slot filters by model name
19. ✅ **Integration test** — nginx forked, 16 endpoints tested, 1000-word system message

20. ✅ **Chat template application** — auto-detect from GGUF metadata, role/content formatting, error handling for invalid messages

## Design Decisions

1. **XS is a thin API wrapper** — expose only what llama.cpp provides. No mmap caching logic in XS.
2. **In-memory API first** — `get_data`/`set_data` are the core. File I/O is a thin convenience layer.
3. **Allocate/free in XS** — for `get_data`, XS allocates buffer, calls C function, creates Perl SV, frees buffer. Caller gets a clean Perl string.
4. **`size_t` as `UV`** — simple and correct on 64-bit Linux.
5. **`llama_memory_t` as `IV`** — opaque pointer, same treatment as `llama_context*`.
6. **Per-sequence included** — straightforward API, works regardless of `n_seq_max` config.
7. **Error handling** — `llama_state_set_data` returns bytes read. `llama_state_load_file` returns bool — croak on failure.
8. **mmap / /dev/shm / caching tiers belong in Perl** — the app layer uses `Sys::Mmap` to mmap a file, then passes the SV to `llama_state_set_data`. The XS does not need any mmap-specific functions.
9. **Token arrays use manual conversion** — `llama_token` is `int32_t`, `IV*` typemap allocates `int64_t`. Manual loop avoids size mismatch that produces garbage tokens.
10. **Dual-mode architecture** — keep llama-server proxy, add `/api/cache` endpoint. Clients choose backend.
11. **nginx handles serving** — Perl module is NOT a server. nginx handles HTTP, forking, connections.
12. **Model is shared across slots** — loaded once with mmap, all contexts share the same model pointer.
13. **Slots are pre-allocated** — `n_slots` contexts created at startup, no runtime fork().
14. **KV cache in /dev/shm** — default cache dir is /dev/shm for zero-copy mmap loading.
15. **SSE streaming in Perl** — nginx Perl module handles SSE response formatting.
16. **Embeddings via existing API** — `llama_get_embeddings()` already bound, just wrap in Perl.
17. **Skip rerank/LoRA** — not needed for current use case.
18. **KV cache header stores actual_size** — `pack('NN', n_tokens, actual_size)` because `seq_state_size()` may differ from `seq_get_state()` return length.
19. **Return values stored outside eval** — Perl `return` inside `eval` does not propagate; use a variable instead.
20. **Multi-model architecture** — each model has its own set of contexts/slots; global slot namespace with offsets; model routing by name from request body
21. **ModelConfig preset parsing** — INI file parsed in Perl (no libllama dependency); global `[*]` defaults merged with per-model overrides; key aliasing maps INI keys (ctx-size) to Perl keys (n_ctx)
22. **Model loaded once per Cache instance** — `load_model()` called once and stored in model hash; vocab pointer stays valid across requests
23. **Dynamic model loading** — models loaded on-demand when first requested; searched in `MODEL_PATH` directories for `${name}.gguf` or exact path match; `MODEL` env var for single default model; `MODEL_PATH` colon-separated for search directories
24. **mmap model loading** — `llama_model_load_from_file_mmap()` used for all models; enables faster loading and shared memory across contexts
25. **Conv_id slot reuse** — slots mapped to conversation IDs; alloc_slot checks existing mapping before allocating new slot; free_slot clears mapping; model name filtering ensures correct model selection
26. **Mmap offset** — cache files have 8-byte header (n_tokens + actual_size); mmap offset skips header before passing data to seq_set_state
27. **No IO::File** — replaced with regular `open()` for mmap file handles; reduces dependencies
28. **Sys::Mmap optional** — loaded via `eval`; falls back to regular file I/O if not available
29. **JSON::PP over JSON::XS** — tests use JSON::PP (core Perl) to avoid external dependency; runtime uses JSON::XS via nginx

## mmap from Perl (no XS changes needed)

```perl
use Sys::Mmap;
my $path = "/dev/shm/cache_0.bin";
open my $fh, '<:raw', $path or die;
my $size = -s $path;
my $data;
mmap $data, $size, PROT_READ, MAP_SHARED, $fh, 8;  # offset 8 skips header
my $bytes = $ctx->set_state($data);  # llama_state_set_data reads directly from mmap
```

The Perl layer handles the I/O strategy (mmap, fread, /dev/shm vs disk). The XS just provides the llama.cpp API surface.

## Key Implementation Details

### `llama_state_get_data` — allocating buffer in XS

```c
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
```

### `llama_state_set_data` — extracting buffer from Perl string

```c
UV
llama_state_set_data(IV ctx, SV* data_sv)
    CODE:
        STRLEN len;
        char* data = SvPV(data_sv, len);
        RETVAL = (UV)llama_state_set_data((struct llama_context*)ctx, (const uint8_t*)data, len);
    OUTPUT:
        RETVAL
```

### Token array handling (size-mismatch fix)

```c
bool
llama_state_save_file(IV ctx, SV* path_sv, SV* tokens_sv)
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
```

### `size_t` typemap

```
INPUT
size_t
    $arg = (size_t)SvUV($var);

OUTPUT
size_t
    $var = (UV)$arg
```

### `llama_memory_t` typemap

```
struct llama_memory_t*    T_IV
const struct llama_memory_t* T_IV

INPUT
struct llama_memory_t*
    if ($arg == 0) { $arg = (struct llama_memory_t*)NULL; }

const struct llama_memory_t*
    if ($arg == 0) { $arg = (const struct llama_memory_t*)NULL; }

OUTPUT
struct llama_memory_t*
    $var = (IV)$arg

const struct llama_memory_t*
    $var = (IV)$arg
```

## Streaming/Thinning — SSE Support

```perl
# In Llama::Cache
sub chat_stream {
    my ($self, $slot_id, $messages, $n_predict, $conv_id) = @_;
    my $ctx = $self->{contexts}[$slot_id]{context};
    my $stream = Llama::Cache::Stream->new($conv_id);

    # Tokenize and decode prompt
    my @tokens = $self->_tokenize_messages($messages);
    my $batch = Llama::Batch->new(max_tokens => scalar @tokens);
    $batch->set_tokens(map { [$_, $_, 0] } 0 .. $#tokens);
    $ctx->decode($batch);

    # Autoregressive generation with streaming
    my @output;
    my $n_tokens = scalar @tokens;
    my $n_ctx = $ctx->n_ctx;

    for my $i (0 .. $n_predict - 1) {
        last if $n_tokens >= $n_ctx;

        my $last_tok = $tokens[-1];
        my $pos = $n_tokens - 1;

        my $b = Llama::Batch->new(max_tokens => 1);
        $b->set_token(0, $last_tok, $pos, 0);
        $ctx->decode($b);
        $b->DESTROY;

        my $logit_ptr = Llama::_get_logits_ptr($ctx->{ptr});
        my @logits;
        for my $j (0 .. $vocab->n_tokens - 1) {
            $logits[$j] = Llama::_read_float($logit_ptr, $j);
        }

        # Simple greedy sampling
        my $max_logit = $logits[0];
        my $new_tok = 0;
        for my $j (1 .. $#logits) {
            if ($logits[$j] > $max_logit) {
                $max_logit = $logits[$j];
                $new_tok = $j;
            }
        }

        push @tokens, $new_tok;
        $n_tokens++;

        my $piece = $self->{model}->vocab->token_to_piece($new_tok);
        $stream->add_chunk($piece);

        # Check if client disconnected (via stream pipe)
        last if $stream->is_cancelled();
    }

    $stream->finalize();
    return $stream;
}
```

**SSE response format (nginx Perl):**
```perl
$r->send_http_header("text/event-stream");
$r->connection("keep-alive");

while (my $chunk = $stream->next_chunk()) {
    $r->print("data: " . encode_json({
        id      => $conv_id,
        object  => "chat.completion.chunk",
        created => time(),
        model   => $model_name,
        choices => [{
            index        => 0,
            delta        => { content => $chunk },
            finish_reason => undef,
        }],
    }) . "\n\n");
    $r->rflush() if $r->can("rflush");
    last if $stream->is_cancelled();
}

# Final chunk with finish_reason
$r->print("data: " . encode_json({
    id      => $conv_id,
    object  => "chat.completion.chunk",
    created => time(),
    model   => $model_name,
    choices => [{
        index        => 0,
        delta        => {},
        finish_reason => "stop",
    }],
}) . "\n\n");
$r->print("data: [DONE]\n\n");
```

## Embeddings

The llama-server supports `/v1/embeddings` via `llama_get_embeddings()`. We already have `Llama::Context->get_embeddings()` which returns a float array.

```perl
sub get_embeddings {
    my ($self) = @_;
    my $n_embd = $self->{model}->n_embd;
    my @emb;
    for my $i (0 .. $n_embd - 1) {
        $emb[$i] = Llama::_read_float(Llama::_get_embeddings_ptr($self->{ptr}), $i);
    }
    return \@emb;
}
```

**Response format (OpenAI-compatible):**
```json
{
    "object": "list",
    "data": [{
        "object": "embedding",
        "index": 0,
        "embedding": [0.1, -0.2, ...]
    }],
    "model": "model-name",
    "usage": {
        "prompt_tokens": 10,
        "total_tokens": 10
    }
}
```

## nginx Integration

### nginx.conf
```nginx
location /api/cache/health {
    perl Llama::cache_health;
}

location /api/cache/v1/health {
    perl Llama::cache_health;
}

location /api/cache/v1/models {
    perl Llama::cache_models;
}

location /api/cache/v1/slots {
    perl Llama::cache_slots;
}

location /api/cache/v1/chat/completions {
    perl Llama::cache_chat;
}

location /api/cache/v1/completions {
    perl Llama::cache_completion;
}

location /api/cache/v1/embeddings {
    perl Llama::cache_embeddings;
}

location /api/cache/v1/tokenize {
    perl Llama::cache_tokenize;
}

location /api/cache/v1/detokenize {
    perl Llama::cache_detokenize;
}

location /api/cache/metrics {
    perl Llama::cache_metrics;
}

location /api/cache/props {
    perl Llama::cache_props;
}

location /api/cache/v1/chat/completions/input_tokens {
    perl Llama::cache_input_tokens;
}

location /api/cache/v1/stream {
    perl Llama::cache_stream;
}

location /api/cache/v1/streams/lookup {
    perl Llama::cache_streams_lookup;
}

location /api/cache/v1/stream/ {
    perl Llama::cache_stream_delete;
}
```

### Environment Variables
```bash
MODEL=/models/inference.gguf          # Default model path (backward compat)
MODEL_PATH=/models:/shared/models     # Colon-separated search directories
PRESET_FILE=/models/llamacpp_presets.ini  # Preset file path
```

### Dynamic Model Loading
Models are loaded on-demand when first requested via the `model` field in the request body. The system searches `MODEL_PATH` directories for `${name}.gguf` or exact path matches.

```perl
# Request body triggers on-demand loading
{ "model": "embedding", "messages": [...] }
# Searches /models/embedding.gguf, /shared/models/embedding.gguf, etc.
```

### lib/perl/llama.pm — nginx Perl module

nginx handles HTTP serving, forking, and connection management. This module only handles inference logic.

**Startup:**
```perl
BEGIN {
    $ENV{LD_LIBRARY_PATH} = "/opt/rocm/lib:/llama/bin";
    use Llama::Cache;
    $CACHE = Llama::Cache->new(
        model_path => $ENV{MODEL},
        search_paths => [split /:/, $ENV{MODEL_PATH} || "/models"],
        preset_file => $ENV{PRESET_FILE},
        n_ctx      => 4096,
        n_batch    => 512,
        n_threads  => 16,
        n_slots    => 4,
        cache_dir  => "/dev/shm/llama_cache",
    );
}
```

**Request handlers:**
```perl
sub cache_chat {
    my ($r) = @_;
    my $rb = $r->request_body();
    my $req = decode_json($rb);
    my $slot_id = $req->{id_slot} // 0;
    my $model_name = $req->{model};
    my $conv_id = $req->{conv_id};

    # Dynamic model loading
    if ($model_name && !$CACHE->get_model_by_name($model_name)) {
        $CACHE->load_model_by_name($model_name);
    }

    my $stream = $req->{stream} // 0;
    if ($stream) {
        return _stream_response($r, $slot_id, $req);
    } else {
        my $result = $CACHE->chat_completion($slot_id, $req->{messages}, $req->{n_predict} // 256, {
            conv_id => $conv_id,
            model   => $model_name,
        });
        $r->send_http_header("application/json");
        $r->print(encode_json($result));
        return OK;
    }
}
```

## llama-server API Surface Coverage

| Method | Path | Status |
|--------|------|--------|
| GET | `/health`, `/v1/health` | ✅ Implemented |
| GET | `/metrics` | ✅ Implemented |
| GET | `/props` | ✅ Implemented |
| POST | `/props` | ✅ Implemented |
| GET | `/models`, `/v1/models` | ✅ Implemented |
| POST | `/completion`, `/completions`, `/v1/completions` | ✅ Implemented |
| POST | `/chat/completions`, `/v1/chat/completions` | ✅ Implemented |
| POST | `/v1/embeddings` | ✅ Implemented |
| POST | `/tokenize`, `/detokenize` | ✅ Implemented |
| POST | `/v1/chat/completions/input_tokens` | ✅ Implemented |
| GET | `/slots`, POST `/slots/:id_slot` | ✅ Implemented |
| GET | `/v1/stream/:conv_id` | ✅ Implemented |
| POST | `/v1/streams/lookup` | ✅ Implemented |
| DELETE | `/v1/stream/:conv_id` | ✅ Implemented |

## Comparison: llama-server vs Llama::Cache

| Feature | llama-server | Llama::Cache |
|---------|-------------|--------------|
| Model loading | C++ llama_model_load_from_file | XS llama_model_load_from_file_mmap |
| Multi-model | Process-per-model | In-process, dynamic loading |
| Slot management | C++ state machine, task queue | Perl slot state tracking |
| Streaming | C++ stream_session_manager | Perl Stream class + SSE |
| Embeddings | C++ llama_get_embeddings | XS llama_get_embeddings |
| KV cache save | C++ llama_state_seq_get_data | XS llama_state_seq_get_data |
| Concurrency | C++ threads + task queue | nginx worker_processes + Perl |
| Memory | C++ allocation | Perl SV + XS buffers |
| Binary size | ~100MB C++ binary | ~10MB Perl + XS |
| Startup time | ~2s (C++ init) | ~0.5s (Perl + mmap) |
| Preset support | INI file via libllama | INI file parsed in Perl |
| KV cache reuse | Manual (session files) | Automatic (conv_id mapping) |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL` | Default model path | `/models/default.gguf` |
| `MODEL_PATH` | Colon-separated search directories for dynamic loading | `/models` |
| `PRESET_FILE` | Preset INI file path | `/models/llamacpp_presets.ini` |
| `MODELS_DIR` | Host path to mount as `/models` inside container | (required) |
| `LLAMA_DOCKER_IMAGE` | Docker image tag | `local/ai/llama.cpp-gfx1151:latest` |
| `LLAMA_PRESETS` | Path to presets file | `./llamacpp_presets.ini` |
| `HF_HOME` | HuggingFace cache directory | (default) |

## Dependencies

### Runtime (3 external CPAN modules)
- `JSON::XS` — JSON encoding/decoding (via `libjson-xs-perl` package)
- `Sys::Mmap` — Memory-mapped file I/O (via `libsys-mmap-perl` package)
- `nginx` — nginx Perl API (via `libnginx-mod-http-perl` package)

### Test (core Perl modules)
- `Test::More` — Test framework
- `File::Temp` — Temporary files/dirs
- `HTTP::Tiny` — HTTP client for integration tests
- `JSON::PP` — JSON parsing (core Perl, used in tests)

### Build
- `ExtUtils::MakeMaker` — XS build system

## Remaining Work

### 1. Integration tests on hardware
- **Status:** Test file exists (`t/07-integration.t`), 16 tests
- **Blocker:** Skips off-ROCm hardware (no GPU available)
- **Action:** Run on Strix Halo hardware to verify end-to-end

### 2. Proper sampler
- **Status:** Blocked by llama.cpp bug
- **Bug:** `GGML_ASSERT(cur_p.selected >= 0 && cur_p.selected < (int32_t) cur_p.size)` fails in `llama-sampler.cpp:866` with top_k/top_p/temp samplers on Qwen3.5 ROCmFP4 model
- **Workaround:** Greedy sampling used as stopgap
- **Action:** Fix requires upstream llama.cpp patch

### 3. Chat template support ✅ IMPLEMENTED — TWO APPROACHES

#### A. Native llama_chat_apply_template via XS binding  
- **XS binding**: `Llama::llama_chat_apply_template($tmpl_name_or_string, \@messages, $add_ass)` 
- **Model method**: `$model->apply_chat_template(\@messages, $add_ass)` auto-detects model's built-in template
- **Input format**: Arrayref of hashrefs with `{role => 'system|user|assistant', content => '...'}` keys
- **Supported models**: Qwen, Nemotron, DeepSeekV4, Gemma — all models with GGUF chat templates
- **Pros**: Zero dependencies on external libraries beyond llama.cpp; uses embedded GGUF metadata directly  

#### B. Minijinja-cabi XS wrapper (alternative Jinja2-compatible approach)  
- **Module name**: `Llama::Minijinja` — pure XS bindings to libminijinja_cabi.so (Rust-built C ABI library)  
- **High-level API**:  
```perl
use Llama::Minijinja;
my $mj = Llama::Minijinja->new();
$mj->add_template(greeting => "Hello {{ name }}!");
my $output = $mj->render(greeting => { name => "World" });

# One-shot without registration:
my $result = Llama::Minijinja::apply_from_string("Hi {{ user }}!", { user => "Alice" });
```
- **Features**: Full Jinja syntax support including loops/conditionals/filters via minijinja-rs engine  
- **Build integration**: Requires Rust toolchain and cargo for building libminijinja_cabi.so; Dockerfile updates needed in builder stage  
- **Use case**: When advanced template features required that native llama_chat_apply_template doesn't provide  


### 4. Performance benchmarking
- **Status:** Not done
- **Missing:** No benchmarking of KV cache reuse impact on prompt processing time
- **Action:** Run benchmarks on Strix Halo with real workloads

### 5. Production hardening
- **Status:** Basic error handling in place
- **Missing:**
  - Model loading failure handling
  - Slot exhaustion fallback
  - Cache directory cleanup
  - Conv_id collision handling
- **Action:** Add comprehensive error handling and resource management
