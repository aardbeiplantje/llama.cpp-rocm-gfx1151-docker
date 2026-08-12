package Llama::Types;

use strict; use warnings;

use Exporter 'import';
our @EXPORT_OK = qw(
    LLAMA_MAX_DIMS
    LLAMA_DEFAULT_N_CTX
    LLAMA_DEFAULT_N_BATCH
    LLAMA_DEFAULT_N_THREADS
    LLAMA_STATE_SEQ_FLAGS_NONE
    LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY
    LLAMA_STATE_SEQ_FLAGS_ON_DEVICE
);

# ggml max dimensions
use constant LLAMA_MAX_DIMS => 4;

# Defaults
use constant LLAMA_DEFAULT_N_CTX    => 2048;
use constant LLAMA_DEFAULT_N_BATCH  => 512;
use constant LLAMA_DEFAULT_N_UBATCH => 256;
use constant LLAMA_DEFAULT_N_THREADS => 16;

# State sequence flags
use constant LLAMA_STATE_SEQ_FLAGS_NONE          => 0;
use constant LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY  => 1;
use constant LLAMA_STATE_SEQ_FLAGS_ON_DEVICE     => 2;

1;
