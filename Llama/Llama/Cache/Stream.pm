package Llama::Cache::Stream;

use strict;
use warnings;

our $VERSION = '0.1.0';

sub new {
    my ($class, $conv_id) = @_;
    return bless {
        conv_id      => $conv_id,
        chunks       => [],
        done         => 0,
        cancelled    => 0,
        start_ts     => time(),
        completed_ts => 0,
        from         => 0,
    }, $class;
}

sub add_chunk {
    my ($self, $chunk) = @_;
    push @{$self->{chunks}}, $chunk;
    return 1;
}

sub is_cancelled {
    my ($self) = @_;
    return $self->{cancelled};
}

sub cancel {
    my ($self) = @_;
    $self->{cancelled} = 1;
}

sub finalize {
    my ($self) = @_;
    $self->{done} = 1;
    $self->{completed_ts} = time();
}

sub is_done {
    my ($self) = @_;
    return $self->{done};
}

sub next_chunk {
    my ($self, $from) = @_;
    $from //= $self->{from};
    $self->{from} = $from + 1;

    return undef if $from >= @{$self->{chunks}};
    return $self->{chunks}[$from];
}

sub total_chunks {
    my ($self) = @_;
    return scalar @{$self->{chunks}};
}

sub dropped_prefix {
    my ($self) = @_;
    return $self->{from};
}

sub started_at {
    my ($self) = @_;
    return $self->{start_ts};
}

sub completed_at {
    my ($self) = @_;
    return $self->{completed_ts};
}

sub conv_id {
    my ($self) = @_;
    return $self->{conv_id};
}

1;
