package Llama::Model;

use strict;
use warnings;

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
