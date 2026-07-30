package Llama::ModelConfig;

use strict;
use warnings;

our $VERSION = '0.1.0';

use Llama::Model;

sub new {
    my ($class, %opts) = @_;
    my $path = $opts{path} or die "path required";
    my $preset_file = $opts{preset_file};
    my $preset_name = $opts{preset_name};
    my $overrides = $opts{overrides} || {};

    my $global_defaults = {};
    my $model_overrides = {};

    if ($preset_file && -f $preset_file) {
        my $presets = _parse_preset_file($preset_file);
        $global_defaults = $presets->{global} || {};
        $model_overrides = $presets->{$preset_name} || {};
        if (!$model_overrides && $preset_name) {
            for my $key (keys %$presets) {
                next if $key eq 'global';
                if (_preset_name_matches($key, $preset_name)) {
                    $model_overrides = $presets->{$key};
                    last;
                }
            }
        }
    }

    my $config = {
        path       => $path,
        preset_name => $preset_name,
        _global    => $global_defaults,
        _overrides => $model_overrides,
        _custom    => $overrides,
    };

    return bless $config, $class;
}

sub model_name {
    my ($self) = @_;
    return $self->{preset_name};
}

sub path {
    my ($self) = @_;
    return $self->{path};
}

sub n_ctx {
    my ($self) = @_;
    return $self->_resolve('ctx-size', 'n_ctx');
}

sub n_batch {
    my ($self) = @_;
    return $self->_resolve('batch-size', 'n_batch');
}

sub n_ubatch {
    my ($self) = @_;
    return $self->_resolve('ubatch-size', 'n_ubatch');
}

sub n_threads {
    my ($self) = @_;
    return $self->_resolve('threads', 'n_threads');
}

sub n_threads_batch {
    my ($self) = @_;
    return $self->_resolve('threads-batch', 'n_threads_batch');
}

sub n_predict {
    my ($self) = @_;
    return $self->_resolve('n-predict', 'n_predict');
}

sub keep {
    my ($self) = @_;
    return $self->_resolve('keep', 'keep');
}

sub flash_attn {
    my ($self) = @_;
    return $self->_resolve_bool('flash-attn', 0);
}

sub swa_full {
    my ($self) = @_;
    return $self->_resolve_bool('swa-full', 0);
}

sub cache_prompt {
    my ($self) = @_;
    return $self->_resolve_bool('cache-prompt', 0);
}

sub cache_ram {
    my ($self) = @_;
    return $self->_resolve('cache-ram', -1);
}

sub cache_reuse {
    my ($self) = @_;
    return $self->_resolve('cache-reuse', 0);
}

sub cache_type_k {
    my ($self) = @_;
    return $self->_resolve('cache-type-k', 'q8_0');
}

sub cache_type_v {
    my ($self) = @_;
    return $self->_resolve('cache-type-v', 'q8_0');
}

sub rope_freq_base {
    my ($self) = @_;
    return $self->_resolve('rope-freq-base', undef);
}

sub rope_freq_scale {
    my ($self) = @_;
    return $self->_resolve('rope-freq-scale', undef);
}

sub rope_scaling_type {
    my ($self) = @_;
    return $self->_resolve('rope-scaling-type', undef);
}

sub yarn_orig_ctx {
    my ($self) = @_;
    return $self->_resolve('yarn-orig-ctx', undef);
}

sub yarn_ext_factor {
    my ($self) = @_;
    return $self->_resolve('yarn-ext-factor', undef);
}

sub temp {
    my ($self) = @_;
    return $self->_resolve('temp', 0.8);
}

sub top_k {
    my ($self) = @_;
    return $self->_resolve('top-k', 64);
}

sub top_p {
    my ($self) = @_;
    return $self->_resolve('top-p', 0.95);
}

sub min_p {
    my ($self) = @_;
    return $self->_resolve('min-p', 0.05);
}

sub repeat_penalty {
    my ($self) = @_;
    return $self->_resolve('repeat-penalty', 1.0);
}

sub repeat_last_n {
    my ($self) = @_;
    return $self->_resolve('repeat-last-n', 64);
}

sub frequency_penalty {
    my ($self) = @_;
    return $self->_resolve('frequency-penalty', 0.0);
}

sub presence_penalty {
    my ($self) = @_;
    return $self->_resolve('presence-penalty', 0.0);
}

sub jinja {
    my ($self) = @_;
    return $self->_resolve_bool('jinja', 0);
}

sub reasoning {
    my ($self) = @_;
    return $self->_resolve_bool('reasoning', 0);
}

sub context_shift {
    my ($self) = @_;
    return $self->_resolve_bool('context-shift', 0);
}

sub kv_unified {
    my ($self) = @_;
    return $self->_resolve_bool('kv-unified', 0);
}

sub no_kv_offload {
    my ($self) = @_;
    return $self->_resolve_bool('no-kv-offload', 0);
}

sub no_warmup {
    my ($self) = @_;
    return $self->_resolve_bool('no-warmup', 0);
}

sub embeddings {
    my ($self) = @_;
    return $self->_resolve_bool('embeddings', 0);
}

sub load_model {
    my ($self) = @_;
    return Llama::model_load_mmap($self->{path});
}

sub apply_to_context_opts {
    my ($self) = @_;
    return (
        n_ctx        => $self->n_ctx,
        n_batch      => $self->n_batch,
        n_ubatch     => $self->n_ubatch,
        n_threads    => $self->n_threads,
        n_threads_batch => $self->n_threads_batch,
        flash_attn   => $self->flash_attn,
        swa_full     => $self->swa_full,
        cache_prompt => $self->cache_prompt,
        cache_ram    => $self->cache_ram,
        cache_reuse  => $self->cache_reuse,
        cache_type_k => $self->cache_type_k,
        cache_type_v => $self->cache_type_v,
        rope_freq_base => $self->rope_freq_base,
        rope_freq_scale => $self->rope_freq_scale,
        rope_scaling_type => $self->rope_scaling_type,
        yarn_orig_ctx => $self->yarn_orig_ctx,
        yarn_ext_factor => $self->yarn_ext_factor,
        context_shift => $self->context_shift,
        kv_unified  => $self->kv_unified,
        no_kv_offload => $self->no_kv_offload,
        no_warmup   => $self->no_warmup,
        embeddings  => $self->embeddings,
    );
}

sub _resolve {
    my ($self, $ini_key, $default) = @_;
    my $val = $self->_get_from_chain($ini_key);
    return defined $val ? $val : $default;
}

sub _resolve_bool {
    my ($self, $ini_key, $default) = @_;
    my $val = $self->_get_from_chain($ini_key);
    return _parse_bool($val) if defined $val;
    return $default;
}

sub _get_from_chain {
    my ($self, $key) = @_;

    my @aliases = _get_key_aliases($key);
    my @keys_to_check = ($key, @aliases);

    for my $k (@keys_to_check) {
        return $self->{_custom}{$k} if exists $self->{_custom}{$k};
    }
    for my $k (@keys_to_check) {
        return $self->{_overrides}{$k} if exists $self->{_overrides}{$k};
    }
    for my $k (@keys_to_check) {
        return $self->{_global}{$k} if exists $self->{_global}{$k};
    }
    return undef;
}

sub _get_key_aliases {
    my ($key) = @_;
    my %map = (
        'ctx-size'       => ['n_ctx'],
        'batch-size'     => ['n_batch'],
        'ubatch-size'    => ['n_ubatch'],
        'threads'        => ['n_threads'],
        'threads-batch'  => ['n_threads_batch'],
        'n-predict'      => ['n_predict'],
        'flash-attn'     => ['flash_attn'],
        'swa-full'       => ['swa_full'],
        'cache-prompt'   => ['cache_prompt'],
        'cache-ram'      => ['cache_ram'],
        'cache-reuse'    => ['cache_reuse'],
        'cache-type-k'   => ['cache_type_k'],
        'cache-type-v'   => ['cache_type_v'],
        'rope-freq-base' => ['rope_freq_base'],
        'rope-freq-scale'=> ['rope_freq_scale'],
        'rope-scaling-type' => ['rope_scaling_type'],
        'yarn-orig-ctx'  => ['yarn_orig_ctx'],
        'yarn-ext-factor'=> ['yarn_ext_factor'],
        'context-shift'  => ['context_shift'],
        'kv-unified'     => ['kv_unified'],
        'no-kv-offload'  => ['no_kv_offload'],
        'no-warmup'      => ['no_warmup'],
    );
    if (exists $map{$key}) {
        return @{$map{$key}};
    }
    return ();
}

sub _parse_bool {
    my ($val) = @_;
    return 1 if lc($val) =~ /^(true|yes|1)$/;
    return 0 if lc($val) =~ /^(false|no|0)$/;
    return 0;
}

sub _preset_name_matches {
    my ($pattern, $name) = @_;
    return $pattern eq $name;
}

sub _parse_preset_file {
    my ($path) = @_;
    open my $fh, '<', $path or return { global => {} };
    my $content = do { local $/; <$fh> };
    close $fh;

    my %result;
    my $current_section = 'global';
    $result{$current_section} = {};

    for my $line (split /\n/, $content) {
        $line =~ s/^\s+|\s+$//g;
        next if $line =~ /^$/;
        next if $line =~ /^[#;]/;

        if ($line =~ /^\[(.+?)\]$/) {
            $current_section = $1;
            $result{$current_section} = {} unless exists $result{$current_section};
            next;
        }

        if ($line =~ /^(\S[\S\s]*?)\s*=\s*(.+?)\s*$/) {
            my ($key, $val) = ($1, $2);
            $val =~ s/^\s+|\s+$//g;
            $result{$current_section}{$key} = $val;
        }
    }

    return \%result;
}

1;
