package Llama::Batch;

use strict; use warnings;

sub new {
    my ($class, %opts) = @_;
    my $max_tokens = $opts{max_tokens} || 2048;
    my $embd       = $opts{embd}       || 0;
    my $n_seq_max  = $opts{n_seq_max}  || 1;
    my $ptr = Llama::llama_batch_init($max_tokens, $embd, $n_seq_max);
    return bless {
        ptr        => $ptr,
        max_tokens => $max_tokens,
    }, $class;
}

sub ptr {
    my ($self) = @_;
    return $self->{ptr};
}

sub max_tokens {
    my ($self) = @_;
    return $self->{max_tokens};
}

sub set_token {
    my ($self, $idx, $token, $pos, $seq_id) = @_;
    $seq_id = [$seq_id // 0] unless ref $seq_id;
    Llama::llama_batch_set_token($self->{ptr}, $idx, $token, $pos, $seq_id);
}

sub set_tokens {
    my ($self, @token_pos_seq) = @_;
    for my $i (0 .. $#token_pos_seq) {
        my ($token, $pos, @seq) = @{$token_pos_seq[$i]};
        $self->set_token($i, $token, $pos, \@seq);
    }
    Llama::llama_batch_set_n_tokens($self->{ptr}, scalar @token_pos_seq);
}

sub DESTROY {
    my ($self) = @_;
    return unless $self;
    return unless exists $self->{ptr};
    my $ptr = delete $self->{ptr};
    Llama::llama_batch_free($ptr);
    return;
}

1;
