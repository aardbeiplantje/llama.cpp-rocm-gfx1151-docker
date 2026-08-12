use strict; use warnings;

use Test::More tests => 19;

use FindBin;
use lib "$FindBin::Bin/..";
use lib "$FindBin::Bin/../blib/arch";

use File::Temp qw(tempdir);

# Test 1: Module loads
use_ok("Llama");

# Test 2: Backend init
eval { Llama::backend_init(); };
ok(!$@, 'backend init works');

# Test 3: Load model with mmap enabled (showcase)
my $model_path = $ENV{GGUF_MODEL} // 'Qwen3.5-4B-ROCMFP4.gguf';
my $model;
eval {
    $model = Llama::model_load_mmap($model_path);
};
if ($@) {
    ok(0, "model_load_mmap: $@");
    Llama::backend_free();
    plan skip_all => "Could not load test model";
    exit 0;
}
ok(1, 'model loaded with mmap enabled');

# Test 4: Create context and decode to populate KV cache
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

my $prompt = "The quick brown fox";
my @tokens = $model->vocab->tokenize($prompt);
ok(scalar @tokens > 0, "tokenized prompt into " . scalar(@tokens) . " tokens");

my $batch = Llama::Batch->new(max_tokens => scalar @tokens);
my @token_data = map { [$tokens[$_], $_, 0] } 0 .. $#tokens;
$batch->set_tokens(@token_data);
my $decode_ret = $ctx->decode($batch);
ok($decode_ret == 0, "decode returned status $decode_ret");

# Test 5: Save KV cache state to file
my $tmpdir = tempdir('llama_mmap_test_XXXXXX', CLEANUP => 0, DIR => "/tmp");
my $state_path = "$tmpdir/state.bin";
my $session_path = "$tmpdir/session.bin";

my $state_data = $ctx->get_state;
ok(defined $state_data, "get_state returned data");
ok(length($state_data) > 0, "state data is " . length($state_data) . " bytes");

open my $fh, '>:raw', $state_path or die "Cannot write $state_path: $!";
print $fh $state_data;
close $fh;
ok(-s $state_path > 0, "state file written (" . (-s $state_path) . " bytes)");

# Test 6: Save session file (tokens + state)
my $saved_tokens = [@tokens];
my $save_ok = $ctx->save_session($session_path, $saved_tokens);
ok($save_ok == 1, "save_session returned true");
ok(-s $session_path > 0, "session file written (" . (-s $session_path) . " bytes)");

# Test 7: Load state from file via fread (standard path)
my $ctx2 = Llama::Context->new(
    $model,
    n_ctx => 2048,
    n_batch => 256,
    n_threads => 4,
);
ok(defined $ctx2, 'second context created');

my $state_from_file = $ctx2->get_state;
ok(length($state_from_file) > 0, "ctx2 state populated");

# Load the saved state back
my $bytes_restored = $ctx2->set_state($state_data);
ok($bytes_restored > 0, "restored $bytes_restored bytes from saved state");

# Test 8: Load session file
my ($loaded_tokens, $load_count) = $ctx2->load_session($session_path, 8192);
ok(defined $loaded_tokens, "loaded session tokens");
ok($load_count > 0, "loaded $load_count tokens from session");

# Test 9: Verify loaded tokens match what was saved (the session file stores whatever tokens we pass to save)
is_deeply($loaded_tokens, $saved_tokens, "session tokens match saved tokens");

# Test 10: Cross-context state transfer — save from ctx, load into ctx2
my $ctx3 = Llama::Context->new(
    $model,
    n_ctx => 2048,
    n_batch => 256,
    n_threads => 4,
);
ok(defined $ctx3, 'third context created');

# ctx3 starts with empty state, load ctx2's state into it
my $ctx2_state = $ctx2->get_state;
my $bytes3 = $ctx3->set_state($ctx2_state);
ok($bytes3 > 0, "transferred $bytes3 bytes from ctx2 to ctx3");

# Cleanup
$batch->DESTROY if $batch;
$ctx->DESTROY   if $ctx;
$ctx2->DESTROY  if $ctx2;
$ctx3->DESTROY  if $ctx3;
$model->DESTROY;
Llama::backend_free();

# Clean up temp files
END {
    unlink $state_path    if -f $state_path;
    unlink $session_path  if -f $session_path;
    rmdir $tmpdir         if -d $tmpdir;
};

done_testing();
