package Llama::Context;

use strict;
use warnings;
use Carp qw(croak);

our $VERSION = '0.1.0';

sub new {
    my ($class, $model, %opts) = @_;
    return unless defined $model;
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
    my $chain = Llama::sampler_chain(
        Llama::top_k_sampler(64),
        Llama::top_p_sampler(0.8),
        Llama::temp_sampler(0.8),
    );
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

# ============================================================================
# KV cache state — save/restore (in-memory binary buffer)
# ============================================================================

sub state_size {
    my ($self) = @_;
    return Llama::llama_state_get_size($self->{ptr});
}

sub get_state {
    my ($self) = @_;
    return Llama::llama_state_get_data($self->{ptr});
}

sub set_state {
    my ($self, $data) = @_;
    return Llama::llama_state_set_data($self->{ptr}, $data);
}

# ============================================================================
# KV cache state — save/load session file
# ============================================================================

sub save_session {
    my ($self, $path, $tokens) = @_;
    return Llama::llama_state_save_file($self->{ptr}, $path, $tokens);
}

sub load_session {
    my ($self, $path, $capacity) = @_;
    $capacity ||= 8192;
    my $success = Llama::llama_state_load_file($self->{ptr}, $path, $capacity);
    croak "load_session failed for $path" unless $success;
    my $count = Llama::llama_state_load_file_count();
    my @tokens;
    for my $i (0 .. $count - 1) {
        push @tokens, Llama::llama_state_load_file_token($i);
    }
    Llama::llama_state_load_file_free();
    return (\@tokens, $count);
}

# ============================================================================
# Per-sequence state — save/restore
# ============================================================================

sub seq_state_size {
    my ($self, $seq_id) = @_;
    return Llama::llama_state_seq_get_size($self->{ptr}, $seq_id);
}

sub seq_get_state {
    my ($self, $seq_id) = @_;
    return Llama::llama_state_seq_get_data($self->{ptr}, $seq_id);
}

sub seq_set_state {
    my ($self, $data, $dest_seq_id) = @_;
    return Llama::llama_state_seq_set_data($self->{ptr}, $data, $dest_seq_id);
}

sub seq_save_session {
    my ($self, $path, $seq_id, $tokens) = @_;
    return Llama::llama_state_seq_save_file($self->{ptr}, $path, $seq_id, $tokens);
}

sub seq_load_session {
    my ($self, $path, $seq_id, $capacity) = @_;
    $capacity ||= 8192;
    my $bytes = Llama::llama_state_seq_load_file($self->{ptr}, $path, $seq_id, $capacity);
    croak "seq_load_session failed for $path" unless $bytes > 0;
    my $count = Llama::llama_state_seq_load_file_count();
    my @tokens;
    for my $i (0 .. $count - 1) {
        push @tokens, Llama::llama_state_seq_load_file_token($i);
    }
    Llama::llama_state_seq_load_file_free();
    return (\@tokens, $bytes);
}

# ============================================================================
# Extended per-sequence state (with flags)
# ============================================================================

sub seq_state_size_ext {
    my ($self, $seq_id, $flags) = @_;
    $flags //= 0;
    return Llama::llama_state_seq_get_size_ext($self->{ptr}, $seq_id, $flags);
}

sub seq_get_state_ext {
    my ($self, $seq_id, $flags) = @_;
    $flags //= 0;
    return Llama::llama_state_seq_get_data_ext($self->{ptr}, $seq_id, $flags);
}

sub seq_set_state_ext {
    my ($self, $data, $dest_seq_id, $flags) = @_;
    $flags //= 0;
    return Llama::llama_state_seq_set_data_ext($self->{ptr}, $data, $dest_seq_id, $flags);
}

# ============================================================================
# KV cache manipulation (memory operations)
# ============================================================================

sub clear_kv {
    my ($self, $data_only) = @_;
    $data_only //= 0;
    my $mem = Llama::llama_get_memory($self->{ptr});
    Llama::llama_memory_clear($mem, $data_only);
}

sub seq_rm {
    my ($self, $seq_id, $p0, $p1) = @_;
    $p0 //= -1;
    $p1 //= -1;
    my $mem = Llama::llama_get_memory($self->{ptr});
    return Llama::llama_memory_seq_rm($mem, $seq_id, $p0, $p1);
}

sub seq_cp {
    my ($self, $seq_id_src, $seq_id_dst, $p0, $p1) = @_;
    $p0 //= -1;
    $p1 //= -1;
    my $mem = Llama::llama_get_memory($self->{ptr});
    Llama::llama_memory_seq_cp($mem, $seq_id_src, $seq_id_dst, $p0, $p1);
}

sub seq_keep {
    my ($self, $seq_id) = @_;
    my $mem = Llama::llama_get_memory($self->{ptr});
    Llama::llama_memory_seq_keep($mem, $seq_id);
}

sub seq_add {
    my ($self, $seq_id, $p0, $p1, $delta) = @_;
    $p0 //= -1;
    $p1 //= -1;
    $delta //= 0;
    my $mem = Llama::llama_get_memory($self->{ptr});
    Llama::llama_memory_seq_add($mem, $seq_id, $p0, $p1, $delta);
}

sub seq_div {
    my ($self, $seq_id, $p0, $p1, $d) = @_;
    $p0 //= -1;
    $p1 //= -1;
    $d //= 2;
    my $mem = Llama::llama_get_memory($self->{ptr});
    Llama::llama_memory_seq_div($mem, $seq_id, $p0, $p1, $d);
}

sub seq_pos_min {
    my ($self, $seq_id) = @_;
    my $mem = Llama::llama_get_memory($self->{ptr});
    return Llama::llama_memory_seq_pos_min($mem, $seq_id);
}

sub seq_pos_max {
    my ($self, $seq_id) = @_;
    my $mem = Llama::llama_get_memory($self->{ptr});
    return Llama::llama_memory_seq_pos_max($mem, $seq_id);
}

sub can_shift {
    my ($self) = @_;
    my $mem = Llama::llama_get_memory($self->{ptr});
    return Llama::llama_memory_can_shift($mem);
}

sub DESTROY {
    my ($self) = @_;
    return unless $self;
    return unless exists $self->{ptr};
    my $ptr = delete $self->{ptr};
    Llama::llama_free($ptr);
    return;
}

1;
