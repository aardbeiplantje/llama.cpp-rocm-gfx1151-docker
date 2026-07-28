# KV Cache Save/Restore — Implementation Plan

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

## What's Missing in Current XS Bindings

Current `Llama.xs` has **zero** KV cache state functions.

### XS Functions to Add

```c
size_t  llama_state_get_size(IV ctx)
SV*     llama_state_get_data(IV ctx)        // returns packed binary string
IV      llama_state_set_data(IV ctx, SV* data)  // returns bytes written
bool    llama_state_save_file(IV ctx, SV* path, IV* tokens, IV n_tokens)
bool    llama_state_load_file(IV ctx, SV* path, IV capacity, IV* tokens_out, IV* count_out)

size_t  llama_state_seq_get_size(IV ctx, IV seq_id)
SV*     llama_state_seq_get_data(IV ctx, IV seq_id)
IV      llama_state_seq_set_data(IV ctx, SV* data, IV seq_id)
size_t  llama_state_seq_save_file(IV ctx, SV* path, IV seq_id, IV* tokens, IV n_tokens)
size_t  llama_state_seq_load_file(IV ctx, SV* path, IV seq_id, IV capacity, IV* tokens_out, IV* count_out)

IV      llama_get_memory(IV ctx)            // returns llama_memory_t as IV
void    llama_memory_clear(IV mem, bool data)
IV      llama_memory_seq_rm(IV mem, IV seq_id, IV p0, IV p1)
void    llama_memory_seq_cp(IV mem, IV src, IV dst, IV p0, IV p1)
void    llama_memory_seq_keep(IV mem, IV seq_id)
void    llama_memory_seq_add(IV mem, IV seq_id, IV p0, IV p1, IV delta)
void    llama_memory_seq_div(IV mem, IV seq_id, IV p0, IV p1, IV d)
IV      llama_memory_seq_pos_min(IV mem, IV seq_id)
IV      llama_memory_seq_pos_max(IV mem, IV seq_id)
IV      llama_memory_can_shift(IV mem)
```

### Typemap Entries to Add

```
size_t                    T_UV
llama_memory_t            T_IV
```

`llama_memory_t` is `typedef struct llama_memory_i *` — treat as opaque pointer like `llama_context*`.
`size_t` maps to Perl `UV` (unsigned, platform-native size).

### Perl Module Changes

#### `Llama::Context` — state methods

```perl
$ctx->state_size                          # returns int
my $data = $ctx->get_state                # returns packed binary string (SV)
my $bytes = $ctx->set_state($data)        # restores state, returns bytes written
$ctx->save_session($path, \@tokens)       # saves to .session file
my @tokens = $ctx->load_session($path, $capacity)  # loads from file
```

#### `Llama::Context` — per-sequence state

```perl
$ctx->seq_state_size($seq_id)
my $data = $ctx->seq_get_state($seq_id)
$ctx->seq_set_state($data, $dest_seq_id)
$ctx->seq_save_session($path, $seq_id, \@tokens)
my @tokens = $ctx->seq_load_session($path, $seq_id, $capacity)
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

## Implementation Order

1. **XS bindings for state get/set** — `llama_state_get_size`, `llama_state_get_data`, `llama_state_set_data`
2. **XS bindings for session file I/O** — `llama_state_save_file`, `llama_state_load_file`
3. **XS bindings for per-sequence state** — `llama_state_seq_*` functions
4. **XS bindings for KV cache manipulation** — `llama_get_memory`, `llama_memory_*` functions
5. **Perl wrapper methods in `Llama::Context`** — OO interface for all above
6. **Tests** — `t/03-kv-cache.t` covering save/restore cycle, session file I/O, per-sequence ops
7. **Update `Llama.pm`** — expose state/session helpers at package level if useful

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
IV
llama_state_set_data(IV ctx, SV* data_sv)
    CODE:
        STRLEN len;
        char* data = SvPV(data_sv, len);
        RETVAL = (IV)llama_state_set_data((struct llama_context*)ctx, (const uint8_t*)data, len);
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

1. **XS is a thin API wrapper** — expose only what llama.cpp provides. No mmap, no caching logic in XS.
2. **In-memory API first** — `get_data`/`set_data` are the core. File I/O is a thin convenience layer.
3. **Allocate/free in XS** — for `get_data`, XS allocates buffer, calls C function, creates Perl SV, frees buffer. Caller gets a clean Perl string.
4. **`size_t` as `UV`** — simple and correct on 64-bit Linux.
5. **`llama_memory_t` as `IV`** — opaque pointer, same treatment as `llama_context*`.
6. **Per-sequence included** — straightforward API, works regardless of `n_seq_max` config.
7. **Error handling** — `llama_state_set_data` returns bytes read. `llama_state_load_file` returns bool — croak on failure.
8. **mmap / /dev/shm / caching tiers belong in Perl** — the app layer uses `Sys::Mmap` to mmap a file, then passes the SV to `llama_state_set_data`. The XS does not need any mmap-specific functions.

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

## Testing Strategy

1. **Save/restore roundtrip** — get state, set it back on same context, verify decode produces same results
2. **Cross-context restore** — save state from context A, restore to context B, verify continuity
3. **Session file I/O** — save to disk, load back, verify tokens and state match
4. **Per-sequence ops** — create multi-sequence batch, save/restore individual sequences
5. **KV cache manipulation** — clear, rm, cp, keep, add, div — verify correct behavior
6. **Memory size sanity** — state size should be proportional to context size and model dimensions
