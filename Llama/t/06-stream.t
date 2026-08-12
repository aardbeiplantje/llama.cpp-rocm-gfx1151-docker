use strict; use warnings;

use Test::More tests => 53;

use FindBin;
use lib "$FindBin::Bin/..";
use lib "$FindBin::Bin/../blib/arch";

# Test 1: Module loads
use_ok("Llama");
use_ok("Llama::Cache::Stream");

# Test 2: Create new stream
my $stream = Llama::Cache::Stream->new("conv-abc-123");
isa_ok($stream, 'Llama::Cache::Stream', 'new creates stream object');

# Test 3: conv_id
is($stream->conv_id, "conv-abc-123", "conv_id returns correct id");

# Test 4: started_at returns a number
my $started = $stream->started_at();
ok(defined $started && $started > 0, "started_at returns positive timestamp");

# Test 5: completed_at is 0 initially
is($stream->completed_at(), 0, "completed_at is 0 before finalize");

# Test 6: is_done is false initially
ok(!$stream->is_done(), "is_done is false initially");

# Test 7: is_cancelled is false initially
ok(!$stream->is_cancelled(), "is_cancelled is false initially");

# Test 8: total_chunks is 0 initially
is($stream->total_chunks(), 0, "total_chunks is 0 initially");

# Test 9: dropped_prefix is 0 initially
is($stream->dropped_prefix(), 0, "dropped_prefix is 0 initially");

# Test 10: next_chunk returns undef when empty
my $chunk = $stream->next_chunk();
ok(!defined $chunk, "next_chunk returns undef when empty");

# Test 11: Add chunks
my $n_chunks = 10;
my $stream2 = Llama::Cache::Stream->new("conv-2");
for my $i (0 .. $n_chunks - 1) {
    my $result = $stream2->add_chunk("chunk-$i");
    ok($result == 1, "add_chunk returns 1 for chunk $i");
}
is($stream2->total_chunks(), $n_chunks, "total_chunks is $n_chunks after adding");

# Test 12: next_chunk returns chunks in order using explicit indices
for my $i (0 .. $n_chunks - 1) {
    my $c = $stream2->next_chunk($i);
    is($c, "chunk-$i", "next_chunk($i) returns chunk-$i");
}

# Test 13: next_chunk returns undef after all chunks consumed (uses explicit index)
$chunk = $stream2->next_chunk(100);
ok(!defined $chunk, "next_chunk(100) returns undef when beyond end");

# Test 14: dropped_prefix reflects how many chunks were consumed
# (uses a fresh stream to avoid from=101 from test 13)
my $stream14 = Llama::Cache::Stream->new("conv-14");
for my $i (0 .. $n_chunks - 1) {
    $stream14->add_chunk("pref-$i");
}
for my $i (0 .. $n_chunks - 1) {
    $stream14->next_chunk($i);
}
is($stream14->dropped_prefix(), $n_chunks, "dropped_prefix is $n_chunks after consuming all");

# Test 15: next_chunk with explicit from parameter and gaps
my $stream3 = Llama::Cache::Stream->new("conv-3");
for my $i (0 .. 4) {
    $stream3->add_chunk("item-$i");
}
is($stream3->next_chunk(0), "item-0", "next_chunk(0) returns first chunk");
is($stream3->next_chunk(2), "item-2", "next_chunk(2) returns third chunk");
is($stream3->dropped_prefix(), 3, "dropped_prefix is 3 after jumping to index 2");

# Test 16: finalize sets done and completed_ts
my $stream4 = Llama::Cache::Stream->new("conv-4");
$stream4->add_chunk("hello");
$stream4->add_chunk(" world");
ok(!$stream4->is_done(), "is_done is false before finalize");
$stream4->finalize();
ok($stream4->is_done(), "is_done is true after finalize");
ok($stream4->completed_at() > 0, "completed_at > 0 after finalize");

# Test 17: cancel sets cancelled flag
my $stream5 = Llama::Cache::Stream->new("conv-5");
ok(!$stream5->is_cancelled(), "is_cancelled is false before cancel");
$stream5->cancel();
ok($stream5->is_cancelled(), "is_cancelled is true after cancel");

# Test 18: Stream with different conv_id
my $stream6 = Llama::Cache::Stream->new("unique-id-99");
is($stream6->conv_id, "unique-id-99", "conv_id returns unique id");
$stream6->add_chunk("test");
is($stream6->total_chunks(), 1, "total_chunks is 1");
is($stream6->next_chunk(0), "test", "next_chunk returns added chunk");

# Test 19: Sequential consumption with auto-increment from
my $stream7 = Llama::Cache::Stream->new("conv-7");
for my $i (0 .. 2) {
    $stream7->add_chunk("seq-$i");
}
is($stream7->next_chunk(), "seq-0", "next_chunk() auto-increments from 0");
is($stream7->next_chunk(), "seq-1", "next_chunk() auto-increments from 1");
is($stream7->next_chunk(), "seq-2", "next_chunk() auto-increments from 2");
is($stream7->dropped_prefix(), 3, "dropped_prefix is 3 after consuming all chunks");
is($stream7->next_chunk(), undef, "next_chunk() returns undef after last chunk");
is($stream7->dropped_prefix(), 4, "dropped_prefix is 4 after extra next_chunk call");

# Test 20: Empty stream finalize
my $stream8 = Llama::Cache::Stream->new("conv-8");
$stream8->finalize();
ok($stream8->is_done(), "is_done is true after finalize on empty stream");
is($stream8->total_chunks(), 0, "total_chunks is 0 on empty finalized stream");

done_testing();
