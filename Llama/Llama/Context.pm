package Llama::Context;

use strict;
use warnings;

our $VERSION = '0.1.0';

sub new {
    my ($class, $model, %opts) = @_;
    my $ptr = Llama::llama_init_from_model(
        $model->{ptr},
        $opts{n_ctx}        || 2048,
        $opts{n_batch}      || 512,
        $opts{n_threads}    || 16,
        $opts{n_threads_batch} || 16,
        $opts{embeddings}   || 0,
    );
    return bless {
        ptr         => $ptr,
        model       => $model,
        freed       => 0,
        samplers    => [],
        n_ctx       => $opts{n_ctx} || 2048,
        n_batch     => $opts{n_batch} || 512,
    }, $class;
}

sub ptr {
    my ($self) = @_;
    return $self->{ptr};
}

sub model {
    my ($self) = @_;
    return $self->{model};
}

sub n_ctx {
    my ($self) = @_;
    return Llama::llama_n_ctx($self->{ptr});
}

sub n_batch {
    my ($self) = @_;
    return Llama::llama_n_batch($self->{ptr});
}

sub n_seq_max {
    my ($self) = @_;
    return Llama::llama_n_seq_max($self->{ptr});
}

sub decode {
    my ($self, $batch) = @_;
    my $ret = Llama::llama_decode($self->{ptr}, $batch->{ptr});
    return $ret;
}

sub encode {
    my ($self, $batch) = @_;
    my $ret = Llama::llama_encode($self->{ptr}, $batch->{ptr});
    return $ret;
}

sub get_logits {
    my ($self) = @_;
    my $n_vocab = $self->{model}->n_vocab;
    my $logits_ptr = Llama::_get_logits_ptr($self->{ptr});
    my @logits;
    for my $i (0 .. $n_vocab - 1) {
        $logits[$i] = Llama::_read_float($logits_ptr, $i);
    }
    return \@logits;
}

sub get_logits_ith {
    my ($self, $i) = @_;
    return Llama::llama_get_logits_ith($self->{ptr}, $i);
}

sub get_embeddings {
    my ($self) = @_;
    my $n_embd = $self->{model}->n_embd;
    my @emb;
    for my $i (0 .. $n_embd - 1) {
        $emb[$i] = Llama::_read_float(Llama::_get_embeddings_ptr($self->{ptr}), $i);
    }
    return \@emb;
}

sub default_sampler {
    my ($self) = @_;
    if (!@{$self->{samplers}}) {
        push @{$self->{samplers}}, Llama::top_k_sampler(64);
        push @{$self->{samplers}}, Llama::top_p_sampler(0.8);
        push @{$self->{samplers}}, Llama::temp_sampler(0.8);
    }
    my $chain = Llama::sampler_chain(@{$self->{samplers}});
    return $chain;
}

sub perf_tkvll {
    my ($self) = @_;
    return Llama::llama_perf_context_tkvll($self->{ptr});
}

sub perf_load {
    my ($self) = @_;
    return Llama::llama_perf_context_load($self->{ptr});
}

sub perf_reset {
    my ($self) = @_;
    Llama::llama_perf_context_reset($self->{ptr});
}

sub DESTROY {
    my ($self) = @_;
    if ($self && !$self->{freed}) {
        Llama::llama_free($self->{ptr});
        $self->{freed} = 1;
    }
}

1;
