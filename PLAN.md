# PLAN.md — Perl XS Module for libllama.so

## Overview

Build a Perl CPAN-style module (`Llama`) that provides XS bindings to [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `libllama.so` shared library. Targeted at AMD Strix Halo (gfx1151) systems with ROCm 7.x and 128GB unified memory.

## Goals

- Low-latency Perl interface to llama.cpp inference
- Full model lifecycle: load, query, create context, tokenize, decode, sample
- Pre-fork worker architecture for concurrent inference
- Tested during Docker build (not just runtime)
- gfx1151 Strix Halo 395+ and future gfx1201 (495) support

## Architecture

### Pre-Fork Worker Pattern

```
                    ┌─────────────────────────────────────────────┐
                    │              Perl (nginx + Lua)              │
                    │  ┌───────────┐  ┌───────────┐  ┌──────────┐│
                    │  │  Slot 0   │  │  Slot 1   │  │  Slot N  ││
                    │  └─────┬─────┘  └─────┬─────┘  └────┬─────┘│
                    └────────┼───────────────┼─────────────┼──────┘
                             │               │             │
                    ┌────────┼───────────────┼─────────────┼──────┐
                    │        │               │             │      │
              ┌─────┴──┐ ┌──┴──────┐  ┌────┴─────┐  ┌────┴────┐
              │  Child  │ │  Child  │  │  Child   │  │  Child  │
              │  (Slot 0)│ │ (Slot 1)│  │ (Slot N) │  │ (Slot N)│
              │          │ │         │  │          │  │         │
              │ llama_   │ │ llama_  │  │ llama_   │  │ llama_  │
              │ init_    │ │ init_   │  │ init_    │  │ init_   │
              │ from_    │ │ from_   │  │ init_    │  │ init_   │
              │ model()  │ │ model() │  │ model()  │  │ model() │
              └──────────┘ └─────────┘  └──────────┘  └─────────┘
```

**Why pre-fork, not threads?**
- ROCm/HIP maintains per-process runtime state (streams, memory allocators, driver handles)
- `fork()` after GPU init is undefined behavior — child inherits stale driver handles
- Model weights (mmap'd GGUF) are COW-shared between parent and children
- Each child creates its own `llama_context*` with independent GPU state
- This matches how `llama.cpp`'s own server handles model slots

**What NOT to do:**
- Don't share `llama_context*` across Perl ithreads
- Don't fork after `llama_init_from_model()` without re-init in child
- Don't rely on COW to share GPU memory allocations

## File Structure

```
Llama/
├── Makefile.PL           # ExtUtils::MakeMaker build config
├── typemap               # Custom typemap for llama.cpp C types
├── Llama.xs              # XS declarations and glue code (~260 lines)
├── Llama.pm              # Perl OO wrapper + high-level API
├── Llama/
│   ├── Types.pm          # Enums and constants
│   ├── Model.pm          # OO wrapper for llama_model
│   ├── Context.pm        # OO wrapper for llama_context
│   ├── Batch.pm          # OO wrapper for llama_batch
│   ├── Vocab.pm          # OO wrapper for llama_vocab + tokenization
│   └── Server.pm         # Pre-fork worker server
├── t/
│   ├── 00-load.t         # Module load + basic API tests
│   └── 01-inference.t    # Full model load + context + batch tests
└── .gitignore
```

## Build System

### Makefile.PL

```perl
use ExtUtils::MakeMaker;

my $rocm_path = $ENV{ROCM_PATH} || '/opt/rocm';
my $llama_src = $ENV{LLAMA_SRC}
    or die "Set LLAMA_SRC to the llama.cpp source directory\n";
my $llama_build = $ENV{LLAMA_BUILD} || "$llama_src/build";

my $cc = $ENV{CXX} || 'g++';

WriteMakefile(
    NAME         => 'Llama',
    VERSION_FROM => 'Llama.pm',
    OBJECT       => '\$(O_FILES)',
    CC           => $cc,
    LD           => $cc,
    OPTIMIZE     => '-O2 -std=c++17',
    LIBS         => [
        "-L$llama_build/bin -lllama",
        "-L$rocm_path/lib -lrocblas -lhipblas -lamdhip64",
    ],
    INC          => "-I$llama_src/include -I$llama_src/ggml/include -I$rocm_path/include",
    DEFINE       => "-DVERSION=\$(VERSION) ",
    clean        => { FILES => 'Llama.so Llama.bs' },
    XSPROTOARG   => '-prototypes',
);
```

**Key points:**
- `CC = g++` / `LD = g++` — must link as C++ for llama.cpp symbols
- Links `rocblas`, `hipblas`, `amdhip64` (ROCm HIP runtime)
- `ROCM_PATH` env var (defaults to `/opt/rocm`)
- `LLAMA_SRC` and `LLAMA_BUILD` env vars for flexibility

### Typemap

Custom typemap maps llama.cpp types to Perl types:

```
TYPEMAP

struct llama_model*            T_IV
const struct llama_model*      T_IV
struct llama_context*          T_IV
const struct llama_context*    T_IV
struct llama_vocab*            T_IV
const struct llama_vocab*      T_IV
struct llama_sampler*          T_IV
const struct llama_sampler*    T_IV
struct llama_batch             T_STRUCT
struct llama_model_params      T_STRUCT
struct llama_context_params    T_STRUCT
llama_token_data_array         T_STRUCT
float*                         T_IV
IV*                            T_IV
llama_token                    T_IV
llama_seq_id                   T_IV
llama_pos                      T_I32
```

**Mapping strategy:**
- Opaque pointers → `IV` (Perl integer, file-descriptor style)
- Structs by value → `T_STRUCT` (copy in/out)
- `float*` → `IV` (caller copies data from pointer)
- `llama_token` → `IV` (32-bit integer)

## XS Bindings

### Lifecycle

```xs
void llama_backend_init()
void llama_backend_free()
```

### Model

```xs
IV llama_model_load_from_file(char* path_model)
void llama_model_free(IV model)
IV llama_model_n_ctx_train(IV model)
IV llama_model_n_embd(IV model)
IV llama_model_n_layer(IV model)
UV llama_model_n_params(IV model)
UV llama_model_size(IV model)
IV llama_model_desc(IV model, char* buf, IV buf_size)
const char* llama_model_chat_template(IV model, char* name)
IV llama_model_get_vocab(IV model)
```

### Context

```xs
IV llama_init_from_model(IV model, IV n_ctx, IV n_batch,
                         IV n_threads, IV n_threads_batch, bool embeddings)
void llama_free(IV ctx)
IV llama_n_ctx(IV ctx)
IV llama_n_batch(IV ctx)
IV llama_n_seq_max(IV ctx)
IV llama_decode(IV ctx, IV batch)
IV llama_encode(IV ctx, IV batch)
float* llama_get_logits(IV ctx)
NV llama_get_logits_ith(IV ctx, IV i)
float* llama_get_embeddings(IV ctx)
NV llama_perf_context_load_ms(IV ctx)
NV llama_perf_context_eval_ms(IV ctx)
void llama_perf_context_reset(IV ctx)
```

### Batch

```xs
IV llama_batch_init(IV n_tokens, IV embd, IV n_seq_max)
void llama_batch_free(IV batch)
```

### Vocab / Tokenization

```xs
IV llama_vocab_n_tokens(IV vocab)
llama_token llama_vocab_bos(IV vocab)
llama_token llama_vocab_eos(IV vocab)
llama_token llama_vocab_eot(IV vocab)
llama_token llama_vocab_nl(IV vocab)
IV llama_tokenize_count(IV vocab, char* text, IV max_tokens,
                        bool add_special, bool parse_special)
IV llama_tokenize_get_token(IV idx)
IV llama_token_to_piece(IV vocab, IV token, char* buf, IV buf_len,
                        IV lstrip, bool special)
IV llama_detokenize(IV vocab, IV* tokens, IV n_tokens, char* buf,
                    IV buf_len, bool remove_special, bool unparse_special)
```

### Sampling

```xs
IV llama_sampler_init_greedy()
IV llama_sampler_init_top_k(IV k)
IV llama_sampler_init_top_p(NV p, IV min_keep)
IV llama_sampler_init_temp(NV t)
IV llama_sampler_init_dist(IV seed)
IV llama_sampler_chain_init()
void llama_sampler_chain_add(IV chain, IV sampler)
IV llama_sampler_chain_get(IV chain, IV i)
IV llama_sampler_chain_n(IV chain)
void llama_sampler_free(IV sampler)
void llama_sampler_chain_free(IV chain)
llama_token llama_sampler_sample(IV sampler, IV ctx, IV idx)
void llama_sampler_reset(IV sampler)
```

## Perl OO API

### High-Level Usage

```perl
use Llama;

# Load model and create context
my $llm = Llama->new(
    '/models/model.gguf',
    n_ctx     => 8192,
    n_batch   => 512,
    n_threads => 16,
);

# Generate text
my $output = $llm->generate(
    "The quick brown fox",
    128,
    $llm->default_sampler,
);
print $output;

# Cleanup (happens automatically via DESTROY)
$llm->DESTROY;
```

### Sampler Helpers

```perl
my $greedy = Llama::greedy_sampler();
my $topk   = Llama::top_k_sampler(64);
my $topp   = Llama::top_p_sampler(0.8);
my $temp   = Llama::temp_sampler(0.8);
my $dist   = Llama::dist_sampler(42);

# Chain samplers
my $chain = Llama::sampler_chain($topk, $topp, $temp);
```

### Pre-Fork Server

```perl
use Llama::Server;

Llama::Server::init(
    model_path => '/models/model.gguf',
    n_workers  => 4,
    n_ctx      => 8192,
    n_batch    => 512,
    n_threads  => 16,
);

# Send decode requests to workers
my $result = Llama::Server::decode(0, "Hello world", 128);
```

## ROCm / Strix Halo Environment

Runtime env vars set by `Llama::setup_rocm_env()`:

```perl
$ENV{HSA_OVERRIDE_GFX_VERSION}     = '11.5.1';   # gfx1151 compatibility
$ENV{GGML_CUDA_ENABLE_UNIFIED_MEMORY} = '1';     # mandatory for ROCm
$ENV{GGML_HIP_FORCE_RS_GPU}         = '1';
$ENV{GGML_HIP_FORCE_KV_GPU}         = '1';
$ENV{GGML_HIP_ALLOC_GRAPH_RESERVE}  = '2048';
$ENV{HSA_FORCE_FINE_GRAIN_PCIE}      = '1';
$ENV{HSA_ENABLE_SDMA}                = '0';      # disabled to avoid transfer faults
```

For future gfx1201 (Ryzen AI 495), change `11.5.1` → `12.0.1`.

## Build & Install

```bash
# Build
cd Llama
ROCM_PATH=/opt/rocm \
LLAMA_SRC=/path/to/llama.cpp \
LLAMA_BUILD=/path/to/llama.cpp/build \
perl Makefile.PL && make

# Test
LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH make test

# Install
make install
```

## Docker Integration

The Dockerfile includes a `perl-builder` stage that:
1. Clones llama.cpp from the builder stage
2. Copies the `Llama/` module from the repo
3. Runs `make test` to verify the module works
4. Installs the module to the system Perl library path

The built module is then `COPY --from=perl-builder` into the runtime stage.

## API Surface (Core Inference)

| Category | Functions |
|----------|-----------|
| Lifecycle | `backend_init`, `backend_free` |
| Model | `model_load_from_file`, `model_free`, `n_ctx_train`, `n_embd`, `n_layer`, `n_params`, `model_size`, `desc`, `chat_template`, `get_vocab` |
| Context | `init_from_model`, `free`, `n_ctx`, `n_batch`, `n_seq_max`, `decode`, `encode`, `get_logits`, `get_logits_ith`, `get_embeddings`, `perf_*` |
| Batch | `batch_init`, `batch_free` |
| Vocab | `vocab_n_tokens`, `vocab_bos/eos/eot/nl`, `tokenize_count`, `tokenize_get_token`, `token_to_piece`, `detokenize` |
| Sampling | `sampler_init_*`, `sampler_chain_*`, `sampler_sample`, `sampler_free` |
| Perf | `perf_context_load_ms`, `perf_context_eval_ms`, `perf_context_reset` |

## What's Out of Scope (Phase 1)

- Sampling with logits filtering (top-k/p/temp applied to logits)
- Session/state save & load (`llama_state_*`)
- LoRA adapters (`llama_adapter_lora_*`)
- KV cache management (`llama_memory_*`)
- Chat template application (`llama_chat_apply_template`)
- C callbacks (logging, progress, abort)
- Training API
- Embedding extraction (partial — `get_embeddings` returns pointer, no copy helper yet)

## Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| C++ linkage in XS | Use `LD = g++` in Makefile.PL |
| ROCm driver state after fork | Pre-fork pattern: init GPU in child, not parent |
| Memory leaks in XS | Perl `DESTROY` calls C free functions |
| Large logits array copy | Copy n_vocab floats (~1MB for 256K vocab) — acceptable per token |
| No C++ exceptions in C API | llama.cpp uses return codes; XS checks and `croak()`s on errors |
| Threading in llama.cpp | Single-threaded per context; parent dispatches to pre-forked children |
| ROCm libs unavailable off-Hardware | Tests skip gracefully; Docker build validates on-target |

## Testing

Two test files:

1. **`t/00-load.t`** — Module load, backend init/free, sampler constructors
2. **`t/01-inference.t`** — Full model load, model queries, context creation, batch creation

Tests require ROCm libs (`libhipblas.so.3`, `librocblas.so.5`, `libamdhip64.so.7`) on the target Strix Halo hardware. Off-target, tests skip with explanatory message.
