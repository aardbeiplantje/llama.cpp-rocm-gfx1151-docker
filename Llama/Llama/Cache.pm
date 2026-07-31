package Llama::Cache;

use strict;
use warnings;

our $VERSION = '0.1.0';

use Llama::Types;
use Llama::Model;
use Llama::Context;
use Llama::Batch;
use Llama::Vocab;
use Llama::ModelConfig;

my $HAVE_MMAP = eval { require Sys::Mmap; Sys::Mmap->import(); 1 };
my $HAVE_IO_FILE = eval { require IO::File; IO::File->import(); 1 };

use Llama;

# ============================================================================
# Initialization
# ============================================================================

sub new {
    my ($class, %opts) = @_;

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

    my $cache_dir = $opts{cache_dir} || "/dev/shm/llama_cache";
    unless (-d $cache_dir) {
        eval { require File::Path; File::Path::make_path($cache_dir) };
    }

    my $search_paths = $opts{search_paths} || [];
    my @models;
    my $total_slots = 0;
    my $slot_offset = 0;

    if (my $models_list = $opts{models}) {
        for my $model_spec (@$models_list) {
            my $model;
            if (ref $model_spec && $model_spec->isa('Llama::ModelConfig')) {
                $model = $model_spec;
            } elsif (ref $model_spec && $model_spec->isa('Llama::Model')) {
                $model = bless({ %$model_spec, _is_model => 1 }, 'Llama::ModelConfig');
                $model->{path} = $model_spec->desc;
            } else {
                my $preset_name = $model_spec->{name} || $model_spec->{path};
                $preset_name =~ s/\.gguf$//i;
                $model = Llama::ModelConfig->new(
                    path        => $model_spec->{path},
                    preset_file => $opts{preset_file},
                    preset_name => $preset_name,
                    overrides   => $model_spec->{overrides} || {},
                );
            }

            my $n_slots = $model_spec->{n_slots} || $opts{n_slots} || 4;
            my $model_name = $model->model_name || $model->path;

            my @contexts;
            for my $i (0 .. $n_slots - 1) {
                my $ctx = Llama::Context->new(
                    $model->load_model,
                    n_ctx        => $model->n_ctx,
                    n_batch      => $model->n_batch,
                    n_threads    => $model->n_threads,
                    n_threads_batch => $model->n_threads_batch,
                    embeddings   => $model->embeddings,
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

            push @models, {
                config     => $model,
                contexts   => \@contexts,
                n_slots    => $n_slots,
                slot_offset => $slot_offset,
                model_name => $model_name,
            };

            $total_slots += $n_slots;
            $slot_offset += $n_slots;
        }
    } elsif (my $model_path = $opts{model_path}) {
        my $n_ctx = $opts{n_ctx} || 4096;
        my $n_batch = $opts{n_batch} || 512;
        my $n_threads = $opts{n_threads} || 16;
        my $n_slots = $opts{n_slots} || 4;
        my $embeddings = $opts{embeddings} || 0;

        my $model = Llama::ModelConfig->new(
            path        => $model_path,
            preset_file => $opts{preset_file},
            preset_name => ($opts{model_name} || $model_path) =~ s/\.gguf$//r,
            overrides   => {
                n_ctx        => $n_ctx,
                n_batch      => $n_batch,
                n_threads    => $n_threads,
            },
        );

        my @contexts;
        for my $i (0 .. $n_slots - 1) {
            my $ctx = Llama::Context->new(
                $model->load_model,
                n_ctx        => $n_ctx,
                n_batch      => $n_batch,
                n_threads    => $n_threads,
                embeddings   => $embeddings,
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

        push @models, {
            config      => $model,
            contexts    => \@contexts,
            n_slots     => $n_slots,
            slot_offset => 0,
            model_name  => $model->model_name || $model_path,
        };

        $total_slots = $n_slots;
    }

    return bless {
        models       => \@models,
        search_paths => $search_paths,
        cache_dir    => $cache_dir,
        total_slots  => $total_slots,
        next_slot    => 0,
        stats        => {
            tokens_total            => 0,
            prompt_tokens_total     => 0,
            completion_tokens_total => 0,
        },
    }, $class;
}

# ============================================================================
# Dynamic model loading
# ============================================================================

sub load_model_by_name {
    my ($self, $name) = @_;
    return $self->get_model_by_name($name) if $self->get_model_by_name($name);

    my $search_paths = $self->{search_paths};
    my $model_path;

    for my $dir (@$search_paths) {
        next unless -d $dir;
        my $gguf = "$dir/${name}.gguf";
        if (-f $gguf) {
            $model_path = $gguf;
            last;
        }
        my $full_path = "$dir/$name";
        if (-f $full_path) {
            $model_path = $full_path;
            last;
        }
    }

    return undef unless $model_path;

    my $preset_name = $name;
    $preset_name =~ s/\.gguf$//i;

    my $model = Llama::ModelConfig->new(
        path        => $model_path,
        preset_file => $self->{preset_file},
        preset_name => $preset_name,
    );

    my $n_slots = $self->{default_n_slots} || 4;
    my $model_name = $model->model_name || $model->path;

    my @contexts;
    for my $i (0 .. $n_slots - 1) {
        my $ctx = Llama::Context->new(
            $model->load_model,
            n_ctx        => $model->n_ctx,
            n_batch      => $model->n_batch,
            n_threads    => $model->n_threads,
            n_threads_batch => $model->n_threads_batch,
            embeddings   => $model->embeddings,
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

    my $slot_offset = $self->{total_slots};
    push @{$self->{models}}, {
        config      => $model,
        contexts    => \@contexts,
        n_slots     => $n_slots,
        slot_offset => $slot_offset,
        model_name  => $model_name,
    };

    $self->{total_slots} += $n_slots;

    return $self->get_model_by_name($model_name);
}

sub get_model_by_name {
    my ($self, $name) = @_;
    for my $m (@{$self->{models}}) {
        return $m if $m->{model_name} eq $name;
    }
    return $self->{models}[0] if @{$self->{models}};
    return undef;
}

sub get_model_by_slot {
    my ($self, $slot_id) = @_;
    for my $m (@{$self->{models}}) {
        if ($slot_id >= $m->{slot_offset} && $slot_id < $m->{slot_offset} + $m->{n_slots}) {
            return $m;
        }
    }
    return undef;
}

sub get_slot {
    my ($self, $slot_id) = @_;
    for my $m (@{$self->{models}}) {
        if ($slot_id >= $m->{slot_offset} && $slot_id < $m->{slot_offset} + $m->{n_slots}) {
            my $local = $slot_id - $m->{slot_offset};
            return $m->{contexts}[$local];
        }
    }
    return undef;
}

sub get_model {
    my ($self) = @_;
    return $self->{models}[0];
}

sub get_models {
    my ($self) = @_;
    return $self->{models};
}

# ============================================================================
# Slot management
# ============================================================================

sub alloc_slot {
    my ($self) = @_;
    for my $m (@{$self->{models}}) {
        for my $i (0 .. $m->{n_slots} - 1) {
            if ($m->{contexts}[$i]{state} eq 'idle') {
                $m->{contexts}[$i]{state} = 'busy';
                $self->_restore_slot_cache($m, $i);
                return $m->{slot_offset} + $i;
            }
        }
    }
    return undef;
}

sub free_slot {
    my ($self, $slot_id) = @_;
    my $m = $self->get_model_by_slot($slot_id);
    return unless $m;
    my $local = $slot_id - $m->{slot_offset};
    $self->_save_slot_cache($m, $local);
    $m->{contexts}[$local]{state} = 'idle';
    $m->{contexts}[$local]{n_tokens} = 0;
    $m->{contexts}[$local]{t_start} = 0;
}

sub get_slots {
    my ($self) = @_;
    my %slots;
    for my $m (@{$self->{models}}) {
        for my $i (0 .. $m->{n_slots} - 1) {
            my $global = $m->{slot_offset} + $i;
            my $ctx = $m->{contexts}[$i];
            $slots{$global} = {
                state    => $ctx->{state},
                n_tokens => $ctx->{n_tokens},
                model    => $m->{model_name},
            };
        }
    }
    return \%slots;
}

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

sub model_desc {
    my ($self) = @_;
    return $self->{models}[0]->{model_name};
}

sub model_n_ctx_train {
    my ($self) = @_;
    my $m = $self->{models}[0];
    return undef unless $m;
    my $loaded = $m->{loaded_model} || do {
        $m->{loaded_model} = $m->{config}->load_model;
        $m->{loaded_model};
    };
    return $loaded->n_ctx_train;
}

sub model_n_embd {
    my ($self) = @_;
    my $m = $self->{models}[0];
    return undef unless $m;
    my $loaded = $m->{loaded_model} || do {
        $m->{loaded_model} = $m->{config}->load_model;
        $m->{loaded_model};
    };
    return $loaded->n_embd;
}

sub model_n_params {
    my ($self) = @_;
    my $m = $self->{models}[0];
    return undef unless $m;
    my $loaded = $m->{loaded_model} || do {
        $m->{loaded_model} = $m->{config}->load_model;
        $m->{loaded_model};
    };
    return $loaded->n_params;
}

sub model_info {
    my ($self) = @_;
    my @info;
    for my $m (@{$self->{models}}) {
        my $loaded = $m->{loaded_model} || do {
            $m->{loaded_model} = $m->{config}->load_model;
            $m->{loaded_model};
        };
        push @info, {
            id       => $m->{model_name},
            object   => "model",
            owned_by => "llama.cpp",
            n_ctx_train => $loaded->n_ctx_train,
            n_embd    => $loaded->n_embd,
            n_params  => $loaded->n_params,
            n_slots   => $m->{n_slots},
        };
    }
    return \@info;
}

# ============================================================================
# Tokenization helpers
# ============================================================================

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

    my $m = $self->get_model_by_slot($slot_id);
    return undef unless $m;
    my $local = $slot_id - $m->{slot_offset};
    my $slot = $m->{contexts}[$local];
    my $ctx = $slot->{context};
    my $model = $m->{config};
    my $loaded_model = $m->{loaded_model} || do {
        $m->{loaded_model} = $model->load_model;
        $m->{loaded_model};
    };
    my $vocab = $loaded_model->vocab;

    my @tokens;
    for my $msg (@$messages) {
        my $content = $msg->{content} // '';
        my @toks = $vocab->tokenize($content);
        push @tokens, @toks;
    }

    my $batch = Llama::Batch->new(max_tokens => scalar @tokens);
    $batch->set_tokens(map { [@tokens[$_], $_, 0] } 0 .. $#tokens);

    my $t0 = time();
    $ctx->decode($batch);
    $slot->{t_prompt} = (time() - $t0) * 1000;
    $slot->{n_tokens} = scalar @tokens;
    $slot->{state} = 'idle';
    $batch->DESTROY;

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

    return {
        id                => "chatcmpl-$slot_id",
        object            => "chat.completion",
        created           => time(),
        model             => $model->model_name,
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
}

# ============================================================================
# Completion (blocking)
# ============================================================================

sub completion {
    my ($self, $slot_id, $prompt, $n_predict, $opts) = @_;
    $n_predict //= 256;
    $opts //= {};

    my $m = $self->get_model_by_slot($slot_id);
    return undef unless $m;
    my $local = $slot_id - $m->{slot_offset};
    my $slot = $m->{contexts}[$local];
    my $ctx = $slot->{context};
    my $model = $m->{config};
    my $loaded_model = $m->{loaded_model} || do {
        $m->{loaded_model} = $model->load_model;
        $m->{loaded_model};
    };
    my $vocab = $loaded_model->vocab;

    my @tokens = $vocab->tokenize($prompt);
    my $batch = Llama::Batch->new(max_tokens => scalar @tokens);
    $batch->set_tokens(map { [@tokens[$_], $_, 0] } 0 .. $#tokens);

    my $t0 = time();
    $ctx->decode($batch);
    $slot->{t_prompt} = (time() - $t0) * 1000;
    $slot->{n_tokens} = scalar @tokens;
    $slot->{state} = 'idle';
    $batch->DESTROY;

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

    return {
        id        => "cmpl-$slot_id",
        object    => "text_completion",
        created   => time(),
        model     => $model->model_name,
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
}

# ============================================================================
# Embeddings
# ============================================================================

sub embeddings {
    my ($self, $slot_id, $input) = @_;

    my $m = $self->get_model_by_slot($slot_id);
    return undef unless $m;
    my $local = $slot_id - $m->{slot_offset};
    my $slot = $m->{contexts}[$local];
    my $ctx = $slot->{context};
    my $model = $m->{config};
    my $loaded_model = $m->{loaded_model} || do {
        $m->{loaded_model} = $model->load_model;
        $m->{loaded_model};
    };
    my $vocab = $loaded_model->vocab;

    my @tokens = $vocab->tokenize($input);
    my $batch = Llama::Batch->new(max_tokens => scalar @tokens);
    $batch->set_tokens(map { [@tokens[$_], $_, 0] } 0 .. $#tokens);

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
# KV cache persistence
# ============================================================================

sub _slot_cache_path {
    my ($self, $arg1, $arg2) = @_;
    my $m;
    my $slot_id;
    
    if (ref $arg1 eq 'HASH' && exists $arg1->{model_name}) {
        $m = $arg1;
        $slot_id = $arg2;
    } else {
        $slot_id = $arg1;
        $m = $self->get_model_by_slot($slot_id);
        return undef unless $m;
    }
    
    my $model_name = $m->{model_name} || "default";
    (my $safe_name = $model_name) =~ s/[^a-zA-Z0-9_-]/_/g;
    return "$self->{cache_dir}/${safe_name}_slot_${slot_id}.cache";
}

sub _save_slot_cache {
    my ($self, $arg1, $arg2) = @_;
    my $m;
    my $slot_id;
    
    if (ref $arg1 eq 'HASH' && exists $arg1->{model_name}) {
        $m = $arg1;
        $slot_id = $arg2;
    } else {
        $slot_id = $arg1;
        $m = $self->get_model_by_slot($slot_id);
        return 0 unless $m;
    }
    
    my $slot = $m->{contexts}[$slot_id];
    my $ctx = $slot->{context};
    return 0 unless $slot->{n_tokens} > 0;

    my $size = $ctx->seq_state_size(0);
    return 0 unless $size > 0;

    my $data = $ctx->seq_get_state(0);
    my $path = $self->_slot_cache_path($m, $slot_id);
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
        warn "[Cache] save_slot_cache($m->{model_name}, $slot_id): $@";
        return 0;
    }
    return -s $path;
}

sub _restore_slot_cache {
    my ($self, $arg1, $arg2) = @_;
    my $m;
    my $slot_id;
    
    if (ref $arg1 eq 'HASH' && exists $arg1->{model_name}) {
        $m = $arg1;
        $slot_id = $arg2;
    } else {
        $slot_id = $arg1;
        $m = $self->get_model_by_slot($slot_id);
        return 0 unless $m;
    }
    
    my $slot = $m->{contexts}[$slot_id];
    my $ctx = $slot->{context};
    my $path = $self->_slot_cache_path($m, $slot_id);
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
        warn "[Cache] restore_slot_cache($m->{model_name}, $slot_id): $@";
        return 0;
    }
}

sub save_slot_to_mmap_file {
    my ($self, $slot_id, $file_path) = @_;
    my $m = $self->get_model_by_slot($slot_id);
    return 0 unless $m;
    my $local = $slot_id - $m->{slot_offset};
    my $slot = $m->{contexts}[$local];
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
    my $m = $self->get_model_by_slot($slot_id);
    return 0 unless $m;
    my $local = $slot_id - $m->{slot_offset};
    my $slot = $m->{contexts}[$local];
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
# Cleanup
# ============================================================================

sub DESTROY {
    my ($self) = @_;
    if ($self) {
        for my $m (@{$self->{models}}) {
            for my $slot (@{$m->{contexts}}) {
                $slot->{context}->DESTROY if $slot->{context};
            }
        }
        Llama::llama_backend_free();
    }
}

1;
