package Llama::Vocab;

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

sub n_tokens {
    my ($self) = @_;
    return Llama::llama_vocab_n_tokens($self->{ptr});
}

sub bos {
    my ($self) = @_;
    return Llama::llama_vocab_bos($self->{ptr});
}

sub eos {
    my ($self) = @_;
    return Llama::llama_vocab_eos($self->{ptr});
}

sub eot {
    my ($self) = @_;
    return Llama::llama_vocab_eot($self->{ptr});
}

sub nl {
    my ($self) = @_;
    return Llama::llama_vocab_nl($self->{ptr});
}

sub token_to_piece {
    my ($self, $token) = @_;
    my $buf = "\0" x 128;
    my $len = Llama::llama_token_to_piece($self->{ptr}, $token, $buf, 128, 0, 0);
    if ($len < 0) {
        $len = -$len;
    }
    return substr($buf, 0, $len);
}

sub tokenize {
    my ($self, $text, $max_tokens) = @_;
    $max_tokens //= length($text) * 2 + 10;
    my $n = Llama::llama_tokenize($self->{ptr}, $text, $max_tokens, 1, 0);
    if ($n < 0) {
        croak("tokenization error: text too long for buffer");
    }
    my @tokens;
    for my $i (0 .. $n - 1) {
        push @tokens, Llama::llama_tokenize_get_token($i);
    }
    return @tokens;
}

sub detokenize {
    my ($self, @tokens) = @_;
    my $buf = "\0" x 512;
    my $n = Llama::llama_detokenize($self->{ptr}, \@tokens, scalar @tokens,
                              $buf, 512, 1, 0);
    if ($n < 0) {
        $n = -$n;
    }
    return substr($buf, 0, $n);
}

sub DESTROY {
    my ($self) = @_;
    # vocab is owned by model, do not free
}

1;
