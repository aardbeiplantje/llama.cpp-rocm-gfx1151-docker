package Llama::Model;

use strict;
use warnings;
use Carp qw(croak);

our $VERSION = '0.1.0';

sub new {
    my ($class, $ptr) = @_;
    return bless { ptr => $ptr, freed => 0 }, $class;
}

sub ptr {
    my ($self) = @_;
    return $self->{ptr};
}

sub n_vocab {
    my ($self) = @_;
    return Llama::llama_model_get_vocab($self->{ptr});
}

sub n_ctx_train {
    my ($self) = @_;
    return Llama::llama_model_n_ctx_train($self->{ptr});
}

sub n_embd {
    my ($self) = @_;
    return Llama::llama_model_n_embd($self->{ptr});
}

sub n_layer {
    my ($self) = @_;
    return Llama::llama_model_n_layer($self->{ptr});
}

sub n_params {
    my ($self) = @_;
    return Llama::llama_model_n_params($self->{ptr});
}

sub model_size {
    my ($self) = @_;
    return Llama::llama_model_size($self->{ptr});
}

sub desc {
    my ($self) = @_;
    my $buf = "\0" x 256;
    Llama::llama_model_desc($self->{ptr}, $buf, 256);
    (split /\0/, $buf)[0];
}

sub chat_template {
    my ($self, $name) = @_;
    $name //= '';
    my $tmpl = Llama::llama_model_chat_template($self->{ptr}, $name);
    return $tmpl // '';
}

sub apply_chat_template {
    my ($self, $messages, $add_ass) = @_;
    croak("messages must be an array reference") unless ref($messages) eq 'ARRAY';
    $add_ass //= 0;

    # Validate message structure and extract template from model if not provided
    for my $msg (@$messages) {
        croak("each message must have 'role' and 'content' keys")
            unless ref($msg) eq 'HASH' && exists($msg->{role}) && exists($msg->{content});
    }

    # Get default template name from model metadata (or use NULL for built-in detection)
    my $template_name = undef;  # Let llama.cpp auto-detect based on model metadata

    return Llama::llama_chat_apply_template($template_name, $messages, $add_ass);
}

sub vocab {
    my ($self) = @_;
    my $vocab_ptr = Llama::llama_model_get_vocab($self->{ptr});
    return Llama::Vocab->new($vocab_ptr);
}

sub DESTROY {
    my ($self) = @_;
    if ($self && !$self->{freed}) {
        Llama::llama_model_free($self->{ptr});
        $self->{freed} = 1;
    }
}

1;
