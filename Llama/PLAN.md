# KV Cache Save/Restore — Implementation Plan

## Status: ✅ COMPLETE

All planned features implemented. 75 tests passing across 5 test files.

## Goal
Add KV cache state save/restore functionality to the Perl XS module, enabling session caching and multi-conversation state management.

## llama.cpp API Summary

### State Save/Restore (in-memory binary buffer)

| Function | Return | Parameters |
|---|---|---|
| `llama_state_get_size(ctx)` | `size_t` | `llama_context* ctx` |
| `llama_state_get_data(ctx, dst, size)` | `size_t` | `ctx`, `uint8_t* dst`, `size_t size` |
| `llama_state_set_data(ctx, src, size)` | `size_t` | `ctx`, `const uint8_t* src`, `size_t size` |

### Session File I/O (on-disk)

| Function | Return | Parameters |
|---|---|---|
| `llama_state_save_file(ctx, path, tokens, n_tokens)` | `bool` | `ctx`, `const char* path`, `const llama_token* tokens`, `size_t n_token_count` |
| `llama_state_load_file(ctx, path, tokens_out, capacity, count_out)` | `bool` | `ctx`, `const char* path`, `llama_token* tokens_out`, `size_t n_token_capacity`, `size_t* n_token_count_out` |

### Per-Sequence State

| Function | Return | Parameters |
|---|---|---|
| `llama_state_seq_get_size(ctx, seq_id)` | `size_t` | `ctx`, `llama_seq_id seq_id` |
| `llama_state_seq_get_data(ctx, dst, size, seq_id)` | `size_t` | `ctx`, `uint8_t* dst`, `size_t size`, `llama_seq_id seq_id` |
| `llama_state_seq_set_data(ctx, src, size, dest_seq_id)` | `size_t` | `ctx`, `const uint8_t* src`, `size_t size`, `llama_seq_id dest_seq_id` |
| `llama_state_seq_save_file(ctx, filepath, seq_id, tokens, n_token_count)` | `size_t` | `ctx`, `const char* filepath`, `llama_seq_id seq_id`, `const llama_token* tokens`, `size_t n_token_count` |
| `llama_state_seq_load_file(ctx, filepath, dest_seq_id, tokens_out, n_token_capacity, n_token_count_out)` | `size_t` | `ctx`, `const char* filepath`, `llama_seq_id dest_seq_id`, `llama_token* tokens_out`, `size_t n_token_capacity`, `size_t* n_token_count_out` |

### Extended Per-Sequence (with flags)

| Function | Return | Parameters |
|---|---|---|
| `llama_state_seq_get_size_ext(ctx, seq_id, flags)` | `size_t` | `ctx`, `llama_seq_id`, `llama_state_seq_flags` |
| `llama_state_seq_get_data_ext(ctx, dst, size, seq_id, flags)` | `size_t` | `ctx`, `uint8_t* dst`, `size_t size`, `llama_seq_id`, `llama_state_seq_flags` |
| `llama_state_seq_set_data_ext(ctx, src, size, dest_seq_id, flags)` | `size_t` | `ctx`, `const uint8_t* src`, `size_t size`, `llama_seq_id`, `llama_state_seq_flags` |

### KV Cache Manipulation

| Function | Return | Parameters |
|---|---|---|
| `llama_get_memory(ctx)` | `llama_memory_t` | `const llama_context* ctx` |
| `llama_memory_clear(mem, data)` | `void` | `llama_memory_t mem`, `bool data` |
| `llama_memory_seq_rm(mem, seq_id, p0, p1)` | `bool` | `mem`, `llama_seq_id`, `llama_pos`, `llama_pos` |
| `llama_memory_seq_cp(mem, src, dst, p0, p1)` | `void` | `mem`, `llama_seq_id src`, `llama_seq_id dst`, `llama_pos`, `llama_pos` |
| `llama_memory_seq_keep(mem, seq_id)` | `void` | `mem`, `llama_seq_id` |
| `llama_memory_seq_add(mem, seq_id, p0, p1, delta)` | `void` | `mem`, `llama_seq_id`, `llama_pos`, `llama_pos`, `llama_pos` |
| `llama_memory_seq_div(mem, seq_id, p0, p1, d)` | `void` | `mem`, `llama_seq_id`, `llama_pos`, `llama_pos`, `int d` |
| `llama_memory_seq_pos_min(mem, seq_id)` | `llama_pos` | `mem`, `llama_seq_id` |
| `llama_memory_seq_pos_max(mem, seq_id)` | `llama_pos` | `mem`, `llama_seq_id` |
| `llama_memory_can_shift(mem)` | `bool` | `llama_memory_t mem` |

## Implemented XS Bindings

All planned bindings implemented. Additional bindings added for model mmap support.

### XS Functions in `Llama.xs`

```c
// Model loading
IV      llama_model_load_from_file(char* path)
IV      llama_model_load_from_file_mmap(char* path)       // NEW: use_mmap=true

// KV cache state (in-memory)
UV      llama_state_get_size(IV ctx)
SV*     llama_state_get_data(IV ctx)                       // returns packed binary string
UV      llama_state_set_data(IV ctx, SV* data_sv)          // returns bytes restored

// KV cache session files (global)
bool    llama_state_save_file(IV ctx, SV* path, SV* tokens_sv)
bool    llama_state_load_file(IV ctx, SV* path, IV capacity)  // tokens stored in global buffer
IV      llama_state_load_file_count()                       // number of tokens loaded
IV      llama_state_load_file_token(IV idx)                 // get ith loaded token
void    llama_state_load_file_free()                        // free loaded token buffer

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

### Typemap Entries in `typemap`

```
size_t                         T_UV
struct llama_memory_t*         T_IV
const struct llama_memory_t*   T_IV
```

`llama_memory_t` is `typedef struct llama_memory_i *` — treated as opaque pointer like `llama_context*`.
`size_t` maps to Perl `UV` (unsigned, platform-native size).

### Key Implementation Detail: Token Array Size Mismatch

**Bug found and fixed**: `llama_token` is `int32_t` but the existing `IV*` typemap allocates `int64_t` arrays. Casting `IV*` to `llama_token*` caused the C code to read 32-bit values from a 64-bit array, producing garbage tokens.

**Fix**: XS functions manually build a correctly-sized `llama_token*` buffer from Perl arrays, avoiding the typemap entirely for token data.

### Perl Module Changes

#### `Llama::Context` — state methods

```perl
$ctx->state_size                          # returns UV
my $data = $ctx->get_state                # returns packed binary string
my $bytes = $ctx->set_state($data)        # restores state, returns UV bytes
$ctx->save_session($path, \@tokens)       # saves to .session file
my ($tokens_ref, $count) = $ctx->load_session($path, $capacity)  # loads from file
```

#### `Llama::Context` — per-sequence state

```perl
$ctx->seq_state_size($seq_id)
my $data = $ctx->seq_get_state($seq_id)
$ctx->seq_set_state($data, $dest_seq_id)
$ctx->seq_save_session($path, $seq_id, \@tokens)
my ($tokens_ref, $bytes) = $ctx->seq_load_session($path, $seq_id, $capacity)
```

#### `Llama::Context` — KV cache management

```perl
$ctx->clear_kv($data_only)
$ctx->seq_rm($seq_id, $p0, $p1)
$ctx->seq_cp($src_seq, $dst_seq, $p0, $p1)
$ctx->seq_keep($seq_id)
my $min = $ctx->seq_pos_min($seq_id)
my $max = $ctx->seq_pos_max($seq_id)
my $can = $ctx->can_shift
```

#### `Llama.pm` — model loading with mmap

```perl
Llama::model_load($path)       # standard load (use_mmap = false)
Llama::model_load_mmap($path)  # load with mmap enabled
```

## Implementation Order

1. ✅ **XS bindings for state get/set** — `llama_state_get_size`, `llama_state_get_data`, `llama_state_set_data`
2. ✅ **XS bindings for session file I/O** — `llama_state_save_file`, `llama_state_load_file`
3. ✅ **XS bindings for per-sequence state** — `llama_state_seq_*` functions
4. ✅ **XS bindings for KV cache manipulation** — `llama_get_memory`, `llama_memory_*` functions
5. ✅ **Perl wrapper methods in `Llama::Context`** — OO interface for all above
6. ✅ **Tests** — `t/03-kv-cache.t` covering save/restore cycle, session file I/O, per-sequence ops
7. ✅ **Model mmap support** — `llama_model_load_from_file_mmap` in XS, `model_load_mmap` in Perl
8. ✅ **Showcase test** — `t/04-mmap-showcase.t` demonstrating mmap model loading + cross-context state transfer

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

## Tests

### Test Files

| File | Tests | Description |
|---|---|---|
| `t/00-load.t` | 1 | Module loads |
| `t/01-model-load.t` | 6 | Model loading, context creation, batch setup |
| `t/02-inference.t` | 11 | Tokenization, decode, logits, sampling |
| `t/03-kv-cache.t` | 30 | State save/restore, session file I/O, per-sequence ops, KV cache manipulation |
| `t/04-mmap-showcase.t` | 19 | Mmap model loading, cross-context state transfer, session roundtrip |

### Test Coverage

1. ✅ **Save/restore roundtrip** — get state, set it back on same context
2. ✅ **Cross-context restore** — save state from context A, restore to context B
3. ✅ **Session file I/O** — save to disk, load back, verify tokens and state match
4. ✅ **Per-sequence ops** — save/restore individual sequences, extended with flags
5. ✅ **KV cache manipulation** — clear, rm, cp, keep, add, div, pos_min/max, can_shift
6. ✅ **Memory size sanity** — state size matches expected values
7. ✅ **Model mmap** — `llama_model_load_from_file_mmap` loads with `use_mmap=true`

### Test Results

```
Files=5, Tests=75,  6 wallclock secs
Result: PASS
```
