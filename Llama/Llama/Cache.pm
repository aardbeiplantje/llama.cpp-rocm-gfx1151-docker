package Llama::Cache;

use strict;
use warnings;

our $VERSION = '0.1.0';

use Llama::Types;
use Llama::Model;
use Llama::Context;
use Llama::Batch;
use Llama::Vocab;

my $HAVE_MMAP = eval { require Sys::Mmap; Sys::Mmap->import(); 1 };
my $HAVE_IO_FILE = eval { require IO::File; IO::File->import(); 1 };

# Load Llama.pm for model_load_mmap and other Perl-level functions
use Llama;

require XSLoader;
XSLoader::load('Llama', $VERSION);

# ============================================================================
# Initialization
# ============================================================================

sub new {
    my ($class, %opts) = @_;
    my $model_path = $opts{model_path} or die "model_path required";
    my $n_ctx      = $opts{n_ctx}      || 4096;
    my $n_batch    = $opts{n_batch}    || 512;
    my $n_threads  = $opts{n_threads}  || 16;
    my $n_slots    = $opts{n_slots}    || 4;
    my $cache_dir  = $opts{cache_dir}  || "/dev/shm/llama_cache";

    # Setup ROCm environment
    unless ($ENV{_LLAMA_ENV_SETUP}) {
        $ENV{HSA_OVERRIDE_GFX_VERSION}     = '11.5.1';
        $ENV{GGML_CUDA_ENABLE_UNIFIED_MEMORY} = '1';
        $ENV{GGML_HIP_FORCE_RS_GPU}         = '1';
        $ENV{GGML_HIP_FORCE_KV_GPU}         = '1';
        $ENV{GGML_HIP_ALLOC_GRAPH_RESERVE}  = '2048';
        $ENV{HSA_FORCE_FINE_GRAIN_PCIE}      = '1';
        $ENV{HSA_ENABLE_SDMA}                = '0';
        $ENV{_LLAMA_ENV_SETUP} = 1;
    }

    Llama::llama_backend_init();

    # Load model with mmap
    my $model = Llama::model_load_mmap($model_path);

    # Create pre-allocated slot contexts
    my @contexts;
    for my $i (0 .. $n_slots - 1) {
        my $ctx = Llama::Context->new(
            $model,
            n_ctx        => $n_ctx,
            n_batch      => $n_batch,
            n_threads    => $n_threads,
            embeddings   => $opts{embeddings} || 0,
        );
        push @contexts, {
            context => $ctx,
            state   => 'idle',
            n_tokens => 0,
            t_start => 0,
            t_prompt => 0,
            t_gen => 0,
        };
    }

    # Create cache directory
    unless (-d $cache_dir) {
        eval { require File::Path; File::Path::make_path($cache_dir) };
    }

    return bless {
        model       => $model,
        contexts    => \@contexts,
        cache_dir   => $cache_dir,
        n_slots     => $n_slots,
        next_slot   => 0,
        stats       => {
            tokens_total            => 0,
            prompt_tokens_total     => 0,
            completion_tokens_total => 0,
        },
    }, $class;
}

# ============================================================================
# Slot management
# ============================================================================

sub alloc_slot {
    my ($self) = @_;
    my @contexts = @{$self->{contexts}};
    for my $i (0 .. $#contexts) {
        if ($contexts[$i]{state} eq 'idle') {
            $self->{contexts}[$i]{state} = 'busy';
            $self->_restore_slot_cache($i);
            return $i;
        }
    }
    return undef;
}

sub free_slot {
    my ($self, $slot_id) = @_;
    return unless defined $self->{contexts}[$slot_id];
    $self->_save_slot_cache($slot_id);
    $self->{contexts}[$slot_id]{state} = 'idle';
    $self->{contexts}[$slot_id]{n_tokens} = 0;
    $self->{contexts}[$slot_id]{t_start} = 0;
}

sub get_slots {
    my ($self) = @_;
    my %slots;
    for my $i (0 .. $self->{n_slots} - 1) {
        my $ctx = $self->{contexts}[$i];
        $slots{$i} = {
            state    => $ctx->{state},
            n_tokens => $ctx->{n_tokens},
        };
    }
    return \%slots;
}

sub get_slot {
    my ($self, $slot_id) = @_;
    return $self->{contexts}[$slot_id];
}

# ============================================================================
# Tokenization helpers
# ============================================================================

sub _tokenize_messages {
    my ($self, $messages) = @_;
    my $vocab = $self->{model}->vocab;
    my @tokens;

    for my $msg (@$messages) {
        my $content = $msg->{content} // '';
        my @toks = $vocab->tokenize($content);
        push @tokens, @toks;
    }

    return \@tokens;
}

sub _build_prompt_batch {
    my ($self, $tokens) = @_;
    my $n_tokens = scalar @$tokens;
    my $batch = Llama::Batch->new(max_tokens => $n_tokens);
    $batch->set_tokens(map { [$tokens->[$_], $_, 0] } 0 .. $n_tokens - 1);
    return $batch;
}

# ============================================================================
# Chat completion (blocking)
# ============================================================================

sub chat_completion {
    my ($self, $slot_id, $messages, $n_predict, $opts) = @_;
    $n_predict //= 256;
    $opts //= {};

    my $slot = $self->{contexts}[$slot_id];
    my $ctx = $slot->{context};
    my $vocab = $self->{model}->vocab;

    # Tokenize messages
    my @tokens = @{$self->_tokenize_messages($messages)};
    my $batch = $self->_build_prompt_batch(\@tokens);

    my $t0 = time();
    $ctx->decode($batch);
    $slot->{t_prompt} = (time() - $t0) * 1000;
    $slot->{n_tokens} = scalar @tokens;
    $slot->{state} = 'idle';
    $batch->DESTROY;

    # Autoregressive generation
    my @output;
    my $n_ctx = $ctx->n_ctx;
    my $t_gen_start = time();

    for my $i (0 .. $n_predict - 1) {
        last if $slot->{n_tokens} >= $n_ctx;

        my $last_tok = $tokens[-1];
        my $pos = $slot->{n_tokens} - 1;

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
        $slot->{n_tokens}++;

        my $piece = $vocab->token_to_piece($new_tok);
        push @output, $piece;
    }

    $slot->{t_gen} = (time() - $t_gen_start) * 1000;
    $slot->{state} = 'idle';

    $self->{stats}{tokens_total} += scalar @tokens;
    $self->{stats}{prompt_tokens_total} += $slot->{n_tokens};
    $self->{stats}{completion_tokens_total} += scalar @output;

    my $result = {
        id                => "chatcmpl-$slot_id",
        object            => "chat.completion",
        created           => time(),
        model             => $self->{model}->desc,
        choices           => [{
            index        => 0,
            message      => { role => "assistant", content => join('', @output) },
            finish_reason => "stop",
        }],
        usage             => {
            prompt_tokens    => $slot->{n_tokens},
            completion_tokens => scalar @output,
            total_tokens     => $slot->{n_tokens} + scalar @output,
        },
        timings           => {
            prompt_n          => $slot->{n_tokens},
            prompt_ms         => $slot->{t_prompt},
            prompt_per_token_ms => $slot->{n_tokens} ? $slot->{t_prompt} / $slot->{n_tokens} : 0,
            prompt_per_second => $slot->{t_prompt} ? $slot->{n_tokens} / ($slot->{t_prompt} / 1000) : 0,
            predicted_n       => scalar @output,
            predicted_ms      => $slot->{t_gen},
            predicted_per_token_ms => scalar @output ? $slot->{t_gen} / scalar @output : 0,
            predicted_per_second => scalar @output ? scalar @output / ($slot->{t_gen} / 1000) : 0,
        },
    };

    return $result;
}

# ============================================================================
# Completion (blocking)
# ============================================================================

sub completion {
    my ($self, $slot_id, $prompt, $n_predict, $opts) = @_;
    $n_predict //= 256;
    $opts //= {};

    my $slot = $self->{contexts}[$slot_id];
    my $ctx = $slot->{context};
    my $vocab = $self->{model}->vocab;

    my @tokens = $vocab->tokenize($prompt);
    my $batch = $self->_build_prompt_batch(\@tokens);

    my $t0 = time();
    $ctx->decode($batch);
    $slot->{t_prompt} = (time() - $t0) * 1000;
    $slot->{n_tokens} = scalar @tokens;
    $slot->{state} = 'idle';
    $batch->DESTROY;

    # Autoregressive generation
    my @output;
    my $n_ctx = $ctx->n_ctx;
    my $t_gen_start = time();

    for my $i (0 .. $n_predict - 1) {
        last if $slot->{n_tokens} >= $n_ctx;

        my $last_tok = $tokens[-1];
        my $pos = $slot->{n_tokens} - 1;

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
        $slot->{n_tokens}++;

        my $piece = $vocab->token_to_piece($new_tok);
        push @output, $piece;
    }

    $slot->{t_gen} = (time() - $t_gen_start) * 1000;
    $slot->{state} = 'idle';

    $self->{stats}{tokens_total} += scalar @tokens;
    $self->{stats}{prompt_tokens_total} += $slot->{n_tokens};
    $self->{stats}{completion_tokens_total} += scalar @output;

    my $result = {
        id        => "cmpl-$slot_id",
        object    => "text_completion",
        created   => time(),
        model     => $self->{model}->desc,
        choices   => [{
            text        => join('', @output),
            index       => 0,
            logprobs    => undef,
            finish_reason => "stop",
        }],
        usage     => {
            prompt_tokens    => $slot->{n_tokens},
            completion_tokens => scalar @output,
            total_tokens     => $slot->{n_tokens} + scalar @output,
        },
    };

    return $result;
}

# ============================================================================
# Embeddings
# ============================================================================

sub embeddings {
    my ($self, $slot_id, $input) = @_;

    my $slot = $self->{contexts}[$slot_id];
    my $ctx = $slot->{context};
    my $vocab = $self->{model}->vocab;

    my @tokens = $vocab->tokenize($input);
    my $batch = $self->_build_prompt_batch(\@tokens);

    $ctx->decode($batch);
    $batch->DESTROY;

    $slot->{n_tokens} = scalar @tokens;
    $slot->{state} = 'idle';

    my $emb = $ctx->get_embeddings();

    $self->{stats}{tokens_total} += scalar @tokens;
    $self->{stats}{prompt_tokens_total} += scalar @tokens;

    return $emb;
}

# ============================================================================
# KV cache persistence — mmap-backed
# ============================================================================

sub _slot_cache_path {
    my ($self, $slot_id) = @_;
    return "$self->{cache_dir}/slot_${slot_id}.cache";
}

sub _save_slot_cache {
    my ($self, $slot_id) = @_;
    my $slot = $self->{contexts}[$slot_id];
    my $ctx = $slot->{context};
    return 0 unless $slot->{n_tokens} > 0;

    my $size = $ctx->seq_state_size(0);
    return 0 unless $size > 0;

    my $data = $ctx->seq_get_state(0);
    my $path = $self->_slot_cache_path($slot_id);
    my $actual_size = length($data);

    my $tmp = "$path.tmp";
    eval {
        open my $fh, '>:raw', $tmp or die "Cannot write $tmp: $!";
        my $header = pack('NN', $slot->{n_tokens}, $actual_size);
        print $fh $header;
        print $fh $data;
        close $fh;
        rename $tmp, $path or die "Cannot rename $tmp to $path: $!";
    };
    if ($@) {
        unlink $tmp if -f $tmp;
        warn "[Cache] save_slot_cache($slot_id): $@";
        return 0;
    }
    return -s $path;
}

sub _restore_slot_cache {
    my ($self, $slot_id) = @_;
    my $slot = $self->{contexts}[$slot_id];
    my $ctx = $slot->{context};
    my $path = $self->_slot_cache_path($slot_id);
    return 0 unless -f $path;

    my $file_size = -s $path;
    return 0 unless $file_size > 8;

    eval {
        open my $fh, '<:raw', $path or die "Cannot read $path: $!";
        my $header;
        my $bytes_read = read $fh, $header, 8;
        die "read $bytes_read header bytes, expected 8" unless defined $bytes_read && $bytes_read == 8;
        my ($n_tokens, $data_size) = unpack('NN', $header);

        my $data;
        $bytes_read = read $fh, $data, $data_size;
        close $fh;
        die "read $bytes_read data bytes, expected $data_size" unless defined $bytes_read && $bytes_read == $data_size;

        my $restored = $ctx->seq_set_state($data, 0);
        $slot->{n_tokens} = $n_tokens;
        return $restored;
    };
    if ($@) {
        warn "[Cache] restore_slot_cache($slot_id): $@";
        return 0;
    }
}

sub save_slot_to_mmap {
    my ($self, $slot_id, $file_path) = @_;
    return 0 unless $HAVE_MMAP;
    my $slot = $self->{contexts}[$slot_id];
    my $ctx = $slot->{context};
    return unless $slot->{n_tokens} > 0;

    my $size = $ctx->seq_state_size(0);
    return unless $size > 0;

    my $data = $ctx->seq_get_state(0);

    eval {
        open my $fh, '>:raw', $file_path or die "Cannot write $file_path: $!";
        print $fh $data;
        close $fh;

        my $mmap_data;
        my $fh2 = IO::File->new($file_path, '<:raw') or die unless $HAVE_IO_FILE;
        my $fsize = -s $file_path;
        my $prot = $HAVE_MMAP ? (PROT_READ() || 1) : 1;
        my $map = $HAVE_MMAP ? (MAP_SHARED() || 1) : 1;
        mmap $mmap_data, $fsize, $prot, $map, $fh2;
        close $fh2;

        my $restored = $ctx->seq_set_state($mmap_data, 0);
        return $restored;
    };
    if ($@) {
        warn "[Cache] save_slot_to_mmap($slot_id): $@";
        return 0;
    }
    return $size;
}

sub load_slot_from_mmap {
    my ($self, $slot_id, $file_path) = @_;
    return 0 unless $HAVE_MMAP;
    my $slot = $self->{contexts}[$slot_id];
    my $ctx = $slot->{context};
    return unless -f $file_path;

    my $file_size = -s $file_path;
    return unless $file_size > 0;

    my $ctx_size = $ctx->seq_state_size(0);
    return unless $ctx_size > 0 && $file_size >= $ctx_size;

    eval {
        open my $fh, '<:raw', $file_path or die "Cannot read $file_path: $!";
        my $data;
        my $bytes_read = read $fh, $data, $ctx_size;
        close $fh;
        die "read $bytes_read bytes, expected $ctx_size" unless defined $bytes_read && $bytes_read == $ctx_size;

        my $restored = $ctx->seq_set_state($data, 0);
        $slot->{n_tokens} = $restored > 0 ? $ctx_size : 0;
        return $restored;
    };
    if ($@) {
        warn "[Cache] load_slot_from_mmap($slot_id): $@";
        return 0;
    }
}

sub save_slot_to_mmap_file {
    my ($self, $slot_id, $file_path) = @_;
    my $slot = $self->{contexts}[$slot_id];
    my $ctx = $slot->{context};
    return 0 unless $slot->{n_tokens} > 0;

    my $size = $ctx->seq_state_size(0);
    return 0 unless $size > 0;

    my $data = $ctx->seq_get_state(0);
    my $actual_size = length($data);

    my $result = 0;
    eval {
        open my $fh, '>:raw', $file_path or die "Cannot write $file_path: $!";
        my $header = pack('NN', $slot->{n_tokens}, $actual_size);
        print $fh $header;
        print $fh $data;
        close $fh;
        $result = -s $file_path;
    };
    if ($@) {
        warn "[Cache] save_slot_to_mmap_file($slot_id): $@";
        return 0;
    }
    return $result;
}

sub load_slot_from_mmap_file {
    my ($self, $slot_id, $file_path) = @_;
    my $slot = $self->{contexts}[$slot_id];
    my $ctx = $slot->{context};
    return 0 unless -f $file_path;

    my $file_size = -s $file_path;
    return 0 unless $file_size > 8;

    my $result = 0;
    eval {
        open my $fh, '<:raw', $file_path or die "Cannot read $file_path: $!";
        my $header;
        my $bytes_read = read $fh, $header, 8;
        die "read $bytes_read header bytes, expected 8" unless defined $bytes_read && $bytes_read == 8;
        my ($n_tokens, $data_size) = unpack('NN', $header);

        my $data;
        $bytes_read = read $fh, $data, $data_size;
        close $fh;
        die "read $bytes_read data bytes, expected $data_size" unless defined $bytes_read && $bytes_read == $data_size;

        $result = $ctx->seq_set_state($data, 0);
        $slot->{n_tokens} = $n_tokens;
    };
    if ($@) {
        warn "[Cache] load_slot_from_mmap_file($slot_id): $@";
        return 0;
    }
    return $result;
}

sub save_slot_cache {
    my ($self, $slot_id, $file_path) = @_;
    return $self->save_slot_to_mmap_file($slot_id, $file_path);
}

sub load_slot_cache {
    my ($self, $slot_id, $file_path) = @_;
    return $self->load_slot_from_mmap_file($slot_id, $file_path);
}

sub list_cached_slots {
    my ($self) = @_;
    my @cached;
    opendir my $dh, $self->{cache_dir} or return @cached;
    while (my $entry = readdir $dh) {
        next unless $entry =~ /\.cache$/;
        push @cached, "$self->{cache_dir}/$entry";
    }
    closedir $dh;
    return @cached;
}

# ============================================================================
# Stats
# ============================================================================

sub get_stats {
    my ($self) = @_;
    return {%{$self->{stats}}};
}

sub reset_stats {
    my ($self) = @_;
    $self->{stats} = {
        tokens_total            => 0,
        prompt_tokens_total     => 0,
        completion_tokens_total => 0,
    };
}

# ============================================================================
# Model info
# ============================================================================

sub model_desc {
    my ($self) = @_;
    return $self->{model}->desc;
}

sub model_n_ctx_train {
    my ($self) = @_;
    return $self->{model}->n_ctx_train;
}

sub model_n_embd {
    my ($self) = @_;
    return $self->{model}->n_embd;
}

sub model_n_params {
    my ($self) = @_;
    return $self->{model}->n_params;
}

# ============================================================================
# Cleanup
# ============================================================================

sub DESTROY {
    my ($self) = @_;
    if ($self) {
        for my $slot (@{$self->{contexts}}) {
            $slot->{context}->DESTROY if $slot->{context};
        }
        $self->{model}->DESTROY if $self->{model};
        Llama::llama_backend_free();
    }
}

1;
