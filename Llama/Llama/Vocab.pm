package Llama::Vocab;

use strict; use warnings;

*n_tokens         = *Llama::llama_vocab_n_tokens;
*bos              = *Llama::llama_vocab_bos;
*eos              = *Llama::llama_vocab_eos;
*eot              = *Llama::llama_vocab_eot;
*nl               = *Llama::llama_vocab_nl;
*token_to_piece   = *Llama::llama_token_to_piece;

sub tokenize {
    my ($self, $text, $max_tokens) = @_;
    $max_tokens //= length($text) * 2 + 10;
    return @{Llama::llama_tokenize($self, $text, $max_tokens, 1, 0)//[]};
}

sub detokenize {
    my ($self, @tokens) = @_;
    return Llama::llama_detokenize($self, \@tokens, scalar @tokens, 1, 0);
}

1;
