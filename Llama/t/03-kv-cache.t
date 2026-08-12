use strict;
use warnings;

use Test::More;
use File::Temp qw(tempfile tempdir);

BEGIN {
    eval { require Llama; };
    if ($@) {
        plan skip_all => "Llama XS module not loadable: $@";
        exit 0;
    }
}

# Test 1: Module loads
ok(1, 'Llama module loaded');

# Test 2: Backend init
eval { Llama::backend_init(); };
ok(!$@, 'backend init works');

# Test 3: Load model
my $model_path = $ENV{GGUF_MODEL} // 'Qwen3.5-4B-ROCMFP4.gguf';
my $model;
eval {
    $model = Llama::model_load($model_path);
};
if ($@) {
    ok(0, "model_load: $@");
    Llama::backend_free();
    plan skip_all => "Could not load test model";
    exit 0;
}
ok(1, 'model loaded successfully');

# Test 4: Create context
my $ctx;
eval {
    $ctx = Llama::Context->new(
        $model,
        n_ctx => 2048,
        n_batch => 256,
        n_threads => 4,
    );
};
if ($@) {
    ok(0, "context init: $@");
    $model->DESTROY;
    Llama::backend_free();
    plan skip_all => "Could not create context";
    exit 0;
}
ok(1, 'context created successfully');

# Test 5: Decode some tokens to populate KV cache
my $prompt = "The quick brown fox";
my @tokens = $model->vocab->tokenize($prompt);
ok(scalar @tokens > 0, "tokenized prompt into " . scalar(@tokens) . " tokens");

my $batch = Llama::Batch->new(max_tokens => scalar @tokens);
my @token_data = map { [$tokens[$_], $_, 0] } 0 .. $#tokens;
$batch->set_tokens(@token_data);
my $decode_ret = $ctx->decode($batch);
ok($decode_ret == 0, "decode returned status $decode_ret");

# Test 6: state_size returns a positive number
my $state_size = $ctx->state_size;
ok($state_size > 0, "state_size = $state_size bytes");

# Test 7: get_state returns binary data
my $state_data = $ctx->get_state;
ok(defined $state_data, "get_state returned defined value");
ok(length($state_data) > 0, "get_state returned " . length($state_data) . " bytes of data");
ok(length($state_data) == $state_size, "state data length matches state_size");

# Test 8: set_state restores state (roundtrip on same context)
my $original_state = $ctx->get_state;
my $bytes_restored = $ctx->set_state($original_state);
ok($bytes_restored > 0, "set_state restored $bytes_restored bytes");

# Test 9: get_state on a second context should produce same size
my $ctx2 = Llama::Context->new(
    $model,
    n_ctx => 2048,
    n_batch => 256,
    n_threads => 4,
);
ok(defined $ctx2, 'second context created');
my $state_size2 = $ctx2->state_size;
ok($state_size2 > 0, "second context state_size = $state_size2");
$ctx2->DESTROY;
undef $ctx2;

# Test 10: save_session saves to file
my $tmpdir = tempdir('llama_session_XXXXXX', CLEANUP => 1);
my $session_path = "$tmpdir/session.bin";
my $saved_tokens = [1, 2, 3, 4, 5];
my $save_result = $ctx->save_session($session_path, $saved_tokens);
ok($save_result == 1, "save_session returned true");
ok(-s $session_path > 0, "session file exists and has content (" . (-s $session_path) . " bytes)");

# Test 11: load_session loads from file
my ($loaded_tokens_ref, $load_count) = $ctx->load_session($session_path, 8192);
ok(defined $loaded_tokens_ref, "load_session returned tokens ref");
ok(ref $loaded_tokens_ref eq 'ARRAY', "loaded tokens is an arrayref");
ok(defined $load_count && $load_count > 0, "load_session read $load_count tokens");

# Test 12: seq_state_size works
my $seq_size = $ctx->seq_state_size(0);
ok($seq_size >= 0, "seq_state_size(0) = $seq_size");

# Test 13: seq_get_state returns data
my $seq_state = $ctx->seq_get_state(0);
ok(defined $seq_state, "seq_get_state returned defined value");

# Test 14: seq_set_state works
my $seq_bytes = $ctx->seq_set_state($seq_state, 0);
ok($seq_bytes > 0, "seq_set_state restored $seq_bytes bytes");

# Test 15: can_shift returns a boolean
my $can_shift = $ctx->can_shift;
ok(defined $can_shift, "can_shift returned defined value");

# Test 16: seq_pos_min and seq_pos_max work after decode
my $pos_min = $ctx->seq_pos_min(0);
my $pos_max = $ctx->seq_pos_max(0);
ok(defined $pos_min, "seq_pos_min returned defined value: $pos_min");
ok(defined $pos_max, "seq_pos_max returned defined value: $pos_max");
ok($pos_max >= $pos_min, "pos_max ($pos_max) >= pos_min ($pos_min)");

# Test 17: clear_kv works
eval { $ctx->clear_kv(0); };
ok(!$@, "clear_kv(0) succeeded");
my $pos_min_after_clear = $ctx->seq_pos_min(0);
ok($pos_min_after_clear < 0, "pos_min after clear_kv is negative (empty cache), got $pos_min_after_clear");

# Test 18: Extended per-sequence state functions work
my $seq_size_ext = $ctx->seq_state_size_ext(0, 0);
ok($seq_size_ext >= 0, "seq_state_size_ext(0, 0) = $seq_size_ext");

my $seq_state_ext = $ctx->seq_get_state_ext(0, 0);
ok(defined $seq_state_ext, "seq_get_state_ext returned defined value");

my $seq_bytes_ext = $ctx->seq_set_state_ext($seq_state_ext, 0, 0);
ok($seq_bytes_ext > 0, "seq_set_state_ext restored $seq_bytes_ext bytes");

# Cleanup
$batch->DESTROY if $batch;
$ctx->DESTROY if $ctx;
$ctx2->DESTROY if $ctx2;
$model->DESTROY;
Llama::backend_free();

# Clean up session file and temp dir
unlink $session_path if defined $session_path && -f $session_path;

done_testing();
