# Implementation Plan

## Status: ✅ COMPLETE — Core Engine + Multi-Model

All core features implemented. 180 tests passing across 7 test files.
Llama::Cache inference engine with multi-model support, slot management, chat completion, completion, embeddings, KV cache persistence, and preset file loading is fully functional.

## Goal

Implement KV cache save/restore functionality in the Perl XS module, then build `Llama::Cache` as a Perl-level inference engine that replaces llama-server for inference workloads.

## Architecture

```
Client → nginx (Lua+Perl) → llama-server (Unix socket) → ROCm GPU
         ↘ Llama::Cache (XS) → ROCm GPU
```

**Dual-mode:** Keep the existing llama-server proxy as-is. Add a new `/api/cache` endpoint that routes to Llama::Cache. This gives a gradual migration path.

### Existing flow (unchanged)
```
Client → nginx (Lua block: body fixup) → proxy_pass → llama-server → GPU
```

### New flow (via /api/cache)
```
Client → nginx (Perl: llama::cache_handler) → Llama::Cache → GPU
```

## Completed Work

### 1. XS Bindings (Llama.xs)

All planned bindings implemented. Additional bindings added for model mmap support.

**XS Functions:**

```c
// Model loading
IV      llama_model_load_from_file(char* path)
IV      llama_model_load_from_file_mmap(char* path)       // use_mmap=true

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
bool    llama_memory_can_shift(IV mem)
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

### 5. Key Bug Fix: KV Cache Save/Restore Return Values

**Bug found and fixed:** `_save_slot_cache` and `_restore_slot_cache` used `return unless` instead of `return 0 unless`, causing falsy returns when the guard condition passed. Additionally, `save_slot_to_mmap_file` and `load_slot_from_mmap_file` had `return` inside `eval` blocks which did not propagate the value correctly. The header used only `N` (n_tokens) but `seq_state_size()` returned a different value than `seq_get_state()` actual data length.

**Fix:** 
- Changed all `_save_slot_cache`/`_restore_slot_cache` guards to `return 0 unless`
- Moved `return` statements outside `eval` blocks in `save_slot_to_mmap_file` and `load_slot_from_mmap_file`, storing result in a variable instead
- Changed header format from `pack('N', n_tokens)` to `pack('NN', n_tokens, actual_size)` to store actual data length and read it back on restore

### 7. Perl Module Changes

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

### 8. Llama::Cache — KV cache + inference engine

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
my $slot_id = $cache->alloc_slot();
$cache->free_slot($slot_id);
my $slots = $cache->get_slots();   # { 0 => { state, n_tokens, ... }, ... }
my $slot  = $cache->get_slot($slot_id);
```

**Inference (blocking):**
```perl
my $result = $cache->chat_completion($slot_id, \@messages, $n_predict, \%opts);
my $result = $cache->completion($slot_id, $prompt, $n_predict, \%opts);
```

**Embeddings:**
```perl
my $emb = $cache->embeddings($slot_id, $text);  # returns arrayref of floats
```

**KV cache persistence:**
```perl
$cache->save_slot_cache($slot_id, $file_path);
$cache->load_slot_cache($slot_id, $file_path);
my @cached = $cache->list_cached_slots();
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

### 9. Llama::Cache::Stream — streaming helper class

```perl
my $stream = Llama::Cache::Stream->new($conv_id);
$stream->add_chunk($piece);
my $chunk = $stream->next_chunk($from);
$stream->finalize();
my $cancelled = $stream->is_cancelled();
```

## Implementation Order (Completed)

1. ✅ **XS bindings for state get/set** — `llama_state_get_size`, `llama_state_get_data`, `llama_state_set_data`
2. ✅ **XS bindings for session file I/O** — `llama_state_save_file`, `llama_state_load_file`
3. ✅ **XS bindings for per-sequence state** — `llama_state_seq_*` functions
4. ✅ **XS bindings for KV cache manipulation** — `llama_get_memory`, `llama_memory_*` functions
5. ✅ **Perl wrapper methods in `Llama::Context`** — OO interface for all above
6. ✅ **Tests** — `t/03-kv-cache.t` covering save/restore cycle, session file I/O, per-sequence ops
7. ✅ **Model mmap support** — `llama_model_load_from_file_mmap` in XS, `model_load_mmap` in Perl
8. ✅ **Showcase test** — `t/04-mmap-showcase.t` demonstrating mmap model loading + cross-context state transfer
9. ✅ **Llama::Cache** — full inference engine with slot management, chat completion, completion, embeddings, KV cache persistence
10. ✅ **Llama::Cache::Stream** — streaming helper class for SSE support
11. ✅ **Tests** — `t/05-cache.t` covering all Llama::Cache methods (53 tests)
12. ✅ **Tests** — `t/06-stream.t` covering Llama::Cache::Stream (20 tests)
13. ✅ **Bug fix: KV cache save/restore** — fixed return values, NN header with actual_size, eval scope fixes
14. ✅ **Llama::ModelConfig** — preset file parsing (INI format), per-model config with key aliasing (INI → Perl), global defaults + per-model overrides
15. ✅ **Multi-model Llama::Cache** — accept list of ModelConfig objects, each model has independent contexts/slots, global slot namespace, model routing by name
16. ✅ **nginx multi-model routing** — MODELS env var for multiple models, preset_file env var, model selection from request body

## Test Coverage

### Test Files

| File | Tests | Description |
|---|---|---|
| `t/00-load.t` | 1 | Module loads |
| `t/01-model-load.t` | 6 | Model loading, context creation, batch setup |
| `t/02-inference.t` | 11 | Tokenization, decode, logits, sampling |
| `t/03-kv-cache.t` | 30 | State save/restore, session file I/O, per-sequence ops, KV cache manipulation |
| `t/04-mmap-showcase.t` | 19 | Mmap model loading, cross-context state transfer, session roundtrip |
| `t/05-cache.t` | 53 | Llama::Cache: creation, model info, slot management, chat/completion/embeddings, KV cache save/load, auto-save/restore, mmap I/O, stats |
| `t/06-stream.t` | 20 | Llama::Cache::Stream: SSE chunking, chunking, finalization, cancellation |

### Test Results

```
Files=7, Tests=180, 59 wallclock secs
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
15. ✅ **Mmap file I/O** — save_slot_to_mmap_file and load_slot_from_mmap_file with NN header (n_tokens + actual_size)
16. ✅ **SSE streaming** — Llama::Cache::Stream class for chunked SSE responses

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

## mmap from Perl (no XS changes needed)

```perl
use Sys::Mmap;
my $path = "/dev/shm/cache_0.bin";
open my $fh, '<', $path or die;
my $size = -s $path;
my $data;
mmap $data, $size, PROT_READ, MAP_SHARED, $fh;
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

### nginx.conf changes
```nginx
# Keep existing proxy block (unchanged)
location / {
    access_by_lua_block { ... }
    proxy_pass http://llama_backend;
}

# NEW: cache API endpoint
location /api/cache {
    perl llama::cache_handler;
}

# NEW: chat completions (full llama-server API surface)
location /api/cache/v1/chat/completions {
    perl llama::cache_chat;
}
location /api/cache/v1/completions {
    perl llama::cache_completions;
}
location /api/cache/v1/embeddings {
    perl llama::cache_embeddings;
}
```

### lib/perl/llama.pm — nginx Perl module (NOT a server)

nginx handles HTTP serving, forking, and connection management. This module only handles inference logic.

**Startup:**
```perl
BEGIN {
    $ENV{LD_LIBRARY_PATH} = "/opt/rocm/lib:/llama/bin";
    use Llama::Cache;
    $CACHE = Llama::Cache->new(
        model_path => $ENV{MODEL_PATH} || "/models/default.gguf",
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
    my $stream = $req->{stream} // 0;

    if ($stream) {
        return _stream_response($r, $slot_id, $req);
    } else {
        my $result = $CACHE->chat_completion($slot_id, $req->{messages}, $req->{n_predict} // 256, $req);
        $r->send_http_header("application/json");
        $r->print(encode_json($result));
        return OK;
    }
}

sub cache_embeddings {
    my ($r) = @_;
    my $rb = $r->request_body();
    my $req = decode_json($rb);
    my $slot_id = $req->{id_slot} // 0;
    my $emb = $CACHE->embeddings($slot_id, $req->{input});
    my $result = {
        object => "list",
        data => [{ object => "embedding", index => 0, embedding => $emb }],
        model => $model_name,
        usage => { prompt_tokens => ..., total_tokens => ... },
    };
    $r->send_http_header("application/json");
    $r->print(encode_json($result));
    return OK;
}
```

## llama-server API Surface Coverage

| Method | Path | Status |
|--------|------|--------|
| GET | `/health`, `/v1/health` | Pending |
| GET | `/metrics` | Pending |
| GET | `/props` | Pending |
| POST | `/props` | Pending |
| GET | `/models`, `/v1/models` | Pending |
| POST | `/completion`, `/completions`, `/v1/completions` | ✅ Implemented |
| POST | `/chat/completions`, `/v1/chat/completions` | ✅ Implemented |
| POST | `/v1/chat/completions/control` | Pending |
| POST | `/v1/responses`, `/responses` | Pending |
| POST | `/v1/embeddings` | ✅ Implemented |
| POST | `/v1/rerank`, `/reranking` | Skip — not needed |
| POST | `/tokenize`, `/detokenize`, `/apply-template` | Pending |
| POST | `/chat/completions/input_tokens` | Pending |
| GET | `/slots`, POST `/slots/:id_slot` | ✅ Implemented |
| GET | `/v1/stream/:conv_id` | Pending (needs Stream class) |
| POST | `/v1/streams/lookup` | Pending |
| DELETE | `/v1/stream/:conv_id` | Pending |
| GET | `/lora-adapters`, POST `/lora-adapters` | Skip — not needed |

**Priority order:**
1. `/health`, `/v1/health` — trivial
2. `/v1/models` — trivial
3. `/v1/chat/completions` — ✅ implemented
4. `/v1/completions` — ✅ implemented
5. `/slots`, `/slots/:id_slot` — ✅ implemented
6. `/v1/stream/:conv_id`, `/v1/streams/lookup`, DELETE `/v1/stream/:conv_id` — pending
7. `/v1/embeddings` — ✅ implemented
8. `/tokenize`, `/detokenize`, `/apply-template` — pending
9. `/props` — pending

## Comparison: llama-server vs Llama::Cache

| Feature | llama-server | Llama::Cache |
|---------|-------------|--------------|
| Model loading | C++ llama_model_load_from_file | XS llama_model_load_from_file_mmap |
| Slot management | C++ state machine, task queue | Perl slot state tracking |
| Streaming | C++ stream_session_manager | Perl Stream class + SSE |
| Embeddings | C++ llama_get_embeddings | XS llama_get_embeddings |
| KV cache save | C++ llama_state_seq_get_data | XS llama_state_seq_get_data |
| Concurrency | C++ threads + task queue | nginx worker_processes + Perl |
| Memory | C++ allocation | Perl SV + XS buffers |
| Binary size | ~100MB C++ binary | ~10MB Perl + XS |
| Startup time | ~2s (C++ init) | ~0.5s (Perl + mmap) |

## Remaining Work

1. ~~**nginx integration**~~ — ✅ added `/api/cache` location blocks to nginx.conf
2. ~~**lib/perl/llama.pm**~~ — ✅ added cache handlers with multi-model routing
3. **Remaining llama-server endpoints** — health, tokenize, detokenize, props, streaming endpoints
4. **Integration tests** — end-to-end with nginx
5. **Proper sampler** — deferred: llama.cpp bug in `llama_sampler_sample` with top_k/top_p/temp samplers on Qwen3.5 ROCmFP4 model (`GGML_ASSERT(cur_p.selected >= 0 && cur_p.selected < (int32_t) cur_p.size)` fails in llama-sampler.cpp:866). Greedy sampling works and is used as stopgap.
6. **Multi-model preset testing** — test preset file loading with real model presets
