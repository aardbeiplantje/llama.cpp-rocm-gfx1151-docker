package Llama;

use strict;
use warnings;

our $VERSION = '0.1.0';

use Llama::Types;
use Llama::Model;
use Llama::Context;
use Llama::Batch;
use Llama::Vocab;

sub dl_load_flags { 0x01 } # RTLD_LAZY

require DynaLoader;
our @ISA = qw(DynaLoader);
__PACKAGE__->bootstrap($VERSION);

# ROCm / Strix Halo environment setup
sub setup_rocm_env {
    return unless $::ROCM_ENV_SETUP;
    $::ROCM_ENV_SETUP = 1;
    $ENV{HSA_OVERRIDE_GFX_VERSION}        = '11.5.1';
    $ENV{GGML_CUDA_ENABLE_UNIFIED_MEMORY} = '1';
    $ENV{GGML_HIP_FORCE_RS_GPU}           = '1';
    $ENV{GGML_HIP_FORCE_KV_GPU}           = '1';
    $ENV{GGML_HIP_ALLOC_GRAPH_RESERVE}    = '2048';
    $ENV{HSA_FORCE_FINE_GRAIN_PCIE}       = '1';
    $ENV{HSA_ENABLE_SDMA}                 = '0';
    return;
}

sub new {
    my ($class, $path, %opts) = @_;
    Llama::setup_rocm_env();
    Llama::backend_init();
    my $model = Llama::model_load($path);
    return unless defined $model;
    my $ctx = Llama::Context->new(
        $model,
        n_ctx           => $opts{n_ctx}           || 2048,
        n_batch         => $opts{n_batch}         || 512,
        n_threads       => $opts{n_threads}       || 16,
        n_threads_batch => $opts{n_threads_batch} || 16,
        embeddings      => $opts{embeddings}      || 0,
    );
    return bless {
        model   => $model,
        context => $ctx,
    }, $class;
}

sub sampler_chain {
    my ($class, @samplers) = @_;
    my $c = Llama::llama_sampler_chain_init();
    Llama::llama_sampler_chain_add($c, $_) for @samplers;
    return $c;
}

sub default_sampler {
    my ($self) = @_;
    return Llama::sampler_chain(
        Llama::sampler_init_top_k(64),
        Llama::sampler_init_top_p(0.8),
        Llama::sampler_init_temp(0.8),
    );
}

sub DESTROY {
    my ($self) = @_;
    return unless $self;
    delete $self->{context};
    delete $self->{model};
    Llama::backend_free();
    return;
}

1;
