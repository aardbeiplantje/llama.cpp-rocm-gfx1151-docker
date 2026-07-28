package Llama;

use strict;
use warnings;

our $VERSION = '0.1.0';

use Llama::Types;
use Llama::Model;
use Llama::Context;
use Llama::Batch;
use Llama::Vocab;

require XSLoader;
XSLoader::load('Llama', $VERSION);

# ROCm / Strix Halo environment setup
sub setup_rocm_env {
    $ENV{HSA_OVERRIDE_GFX_VERSION}     = '11.5.1';
    $ENV{GGML_CUDA_ENABLE_UNIFIED_MEMORY} = '1';
    $ENV{GGML_HIP_FORCE_RS_GPU}         = '1';
    $ENV{GGML_HIP_FORCE_KV_GPU}         = '1';
    $ENV{GGML_HIP_ALLOC_GRAPH_RESERVE}  = '2048';
    $ENV{HSA_FORCE_FINE_GRAIN_PCIE}      = '1';
    $ENV{HSA_ENABLE_SDMA}                = '0';
}

sub backend_init {
    Llama::llama_backend_init();
}

sub backend_free {
    Llama::llama_backend_free();
}

sub model_load {
    my $path = shift;
    my $ptr = Llama::llama_model_load_from_file($path);
    return Llama::Model->new($ptr);
}

sub new {
    my ($class, $path, %opts) = @_;
    setup_rocm_env() unless $ENV{_LLAMA_ENV_SETUP};
    $ENV{_LLAMA_ENV_SETUP} = 1;
    backend_init();
    my $model = model_load($path);
    my $ctx = Llama::Context->new(
        $model,
        n_ctx        => $opts{n_ctx}        || 2048,
        n_batch      => $opts{n_batch}       || 512,
        n_threads    => $opts{n_threads}     || 16,
        n_threads_batch => $opts{n_threads_batch} || 16,
        embeddings   => $opts{embeddings}    || 0,
    );
    my $self = bless {
        model   => $model,
        context => $ctx,
    }, $class;
    return $self;
}

sub generate {
    my ($self, $prompt, $n_predict, $sampler_chain) = @_;
    my $vocab = $self->{model}->vocab;
    my $ctx   = $self->{context};

    my @tokens = $vocab->tokenize($prompt);
    push @tokens, 1; # BOS if not already included

    my $batch = Llama::Batch->new(max_tokens => scalar @tokens);
    for my $i (0 .. $#tokens) {
        $batch->set_token($i, $tokens[$i], $i, 0);
    }

    $ctx->decode($batch);

    my @output;
    my $n_ctx = $ctx->n_ctx;

    for my $i (0 .. $n_predict - 1) {
        my $n_tokens = scalar @tokens;
        if ($n_tokens >= $n_ctx) {
            warn "context full at $n_tokens tokens, stopping";
            last;
        }

        my $last_tok = $tokens[-1];
        my $pos = $n_tokens - 1;

        if ($last_tok >= 0) {
            my $b = Llama::Batch->new(max_tokens => 1);
            $b->set_token(0, $last_tok, $pos, 0);
            $ctx->decode($b);
            $b->DESTROY;
        }

        my $sampler = $sampler_chain || $ctx->default_sampler;
        my $new_tok = Llama::llama_sampler_sample($sampler->{ptr}, $ctx->{ptr}, 0);
        push @tokens, $new_tok;

        my $piece = $vocab->token_to_piece($new_tok);
        push @output, $piece;
    }

    $batch->DESTROY;
    return join('', @output);
}

sub top_k_sampler {
    my ($class, $k) = @_;
    $k //= 64;
    return { ptr => Llama::llama_sampler_init_top_k($k) };
}

sub top_p_sampler {
    my ($class, $p, $min_keep) = @_;
    $p //= 0.8;
    $min_keep //= 1;
    return { ptr => Llama::llama_sampler_init_top_p($p, $min_keep) };
}

sub temp_sampler {
    my ($class, $t) = @_;
    $t //= 0.8;
    return { ptr => Llama::llama_sampler_init_temp($t) };
}

sub dist_sampler {
    my ($class, $seed) = @_;
    $seed //= 42;
    return { ptr => Llama::llama_sampler_init_dist($seed) };
}

sub greedy_sampler {
    return { ptr => Llama::llama_sampler_init_greedy() };
}

sub sampler_chain {
    my ($class, @samplers) = @_;
    my $chain = { ptr => Llama::llama_sampler_chain_init() };
    for my $s (@samplers) {
        Llama::llama_sampler_chain_add($chain->{ptr}, $s->{ptr});
    }
    return $chain;
}

sub DESTROY {
    my ($self) = @_;
    if ($self) {
        $self->{context}->DESTROY if $self->{context};
        $self->{model}->DESTROY   if $self->{model};
        backend_free();
    }
}

1;
