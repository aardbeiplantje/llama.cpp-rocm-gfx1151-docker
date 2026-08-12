package Llama::Model;

use strict; use warnings;

*n_vocab      = *Llama::llama_model_n_get_vocab;
*n_ctx_train  = *Llama::llama_model_n_ctx_train;
*n_embd       = *Llama::llama_model_n_embd;
*n_layer      = *Llama::llama_model_n_layer;
*n_params     = *Llama::llama_model_n_params;
*n_model_size = *Llama::llama_model_n_model_size;
*desc         = *Llama::llama_model_desc;

sub chat_template {
    my ($self, $name) = @_;
    return Llama::llama_model_chat_template($self, $name // "") // "";
}

sub apply_chat_template {
    my ($self, $messages, $add_ass) = @_;
    return Llama::llama_chat_apply_template(undef, $messages, $add_ass // 0);
}

sub vocab {
    my ($self) = @_;
    my $vocab_ptr = Llama::llama_model_get_vocab($self);
    return Llama::Vocab->new($vocab_ptr);
}

sub DESTROY {
    my ($self) = @_;
    return unless $self;
    Llama::llama_model_free($self);
    return;
}

1;
