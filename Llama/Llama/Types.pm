package Llama::Types;

use strict;
use warnings;

our $VERSION = '0.1.0';

use Exporter 'import';
our @EXPORT_OK = qw(
    LLAMA_MAX_DIMS
    LLAMA_DEFAULT_N_CTX
    LLAMA_DEFAULT_N_BATCH
    LLAMA_DEFAULT_N_THREADS
);

# ggml max dimensions
use constant LLAMA_MAX_DIMS => 4;

# Defaults
use constant LLAMA_DEFAULT_N_CTX    => 2048;
use constant LLAMA_DEFAULT_N_BATCH  => 512;
use constant LLAMA_DEFAULT_N_UBATCH => 256;
use constant LLAMA_DEFAULT_N_THREADS => 16;

1;
