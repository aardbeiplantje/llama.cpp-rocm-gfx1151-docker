use strict;
use warnings;

$ENV{LD_LIBRARY_PATH} = '/opt/rocm/lib' unless $ENV{LD_LIBRARY_PATH};

use Test::More;
use File::Temp qw(tempdir);

BEGIN {
    $ENV{PERL5LIB} = 'blib/lib:blib/arch' unless $ENV{PERL5LIB};
    eval { require Llama::Cache; };
    if ($@) {
        plan skip_all => "Llama::Cache not loadable: $@";
        exit 0;
    }
}

# Test 1: Module loads
ok(1, 'Llama::Cache module loaded');

# Test 2: Create cache with model
my $model_path = $ENV{GGUF_MODEL} // '/workdir/llama.cpp.git/llama.cpp/build/bin/Qwen3.5-4B-ROCMFP4.gguf';
my $cache;
eval {
    $cache = Llama::Cache->new(
        model_path => $model_path,
        n_ctx      => 2048,
        n_batch    => 256,
        n_threads  => 4,
        n_slots    => 2,
        cache_dir  => tempdir('cache_XXXXXX', CLEANUP => 0),
    );
};
if ($@) {
    ok(0, "cache init: $@");
    plan skip_all => "Could not create cache";
    exit 0;
}
ok(1, 'cache created successfully');

# Test 3: Model info
my $desc = $cache->model_desc;
ok(length($desc) > 0, "model desc: $desc");

my $n_ctx_train = $cache->model_n_ctx_train;
ok($n_ctx_train > 0, "n_ctx_train = $n_ctx_train");

my $n_embd = $cache->model_n_embd;
ok($n_embd > 0, "n_embd = $n_embd");

my $n_params = $cache->model_n_params;
ok($n_params > 0, "n_params = $n_params");

# Test 4: Slot management
my $slot_0 = $cache->alloc_slot();
ok(defined $slot_0, "alloc_slot returned $slot_0");

my $slot_1 = $cache->alloc_slot();
ok(defined $slot_1, "alloc_slot returned $slot_1");

my $slot_busy = $cache->alloc_slot();
ok(!defined $slot_busy, "alloc_slot returns undef when all slots busy");

my $slots = $cache->get_slots();
ok(ref $slots eq 'HASH', "get_slots returns hashref");
ok(defined $slots->{0}, "slot 0 exists in slots");
ok(defined $slots->{1}, "slot 1 exists in slots");

# Test 5: Chat completion (blocking)
my $slot_id = 0;
my $messages = [
    { role => "user", content => "The quick brown fox" },
];

my $result = $cache->chat_completion($slot_id, $messages, 16);
ok(defined $result, "chat_completion returned result");
ok(defined $result->{id}, "result has id");
ok(defined $result->{choices}, "result has choices");
ok(scalar @{$result->{choices}} > 0, "result has at least one choice");
ok(defined $result->{choices}[0]{message}{content}, "choice has message content");
ok(defined $result->{usage}, "result has usage");
ok($result->{usage}{prompt_tokens} > 0, "usage has prompt_tokens > 0");
ok($result->{usage}{completion_tokens} > 0, "usage has completion_tokens > 0");
ok(defined $result->{timings}, "result has timings");
ok($result->{timings}{prompt_ms} >= 0, "timings has prompt_ms >= 0");
ok($result->{timings}{predicted_ms} >= 0, "timings has predicted_ms >= 0");

# Test 6: Slot state after completion
my $slot_info = $cache->get_slot($slot_id);
ok(defined $slot_info, "get_slot returned slot info");
ok($slot_info->{n_tokens} > 0, "slot n_tokens > 0 after completion");

# Test 7: Completion (blocking)
my $comp_result = $cache->completion(1, "The quick brown fox", 16);
ok(defined $comp_result, "completion returned result");
ok(defined $comp_result->{choices}, "completion has choices");
ok($comp_result->{usage}{prompt_tokens} > 0, "completion usage has prompt_tokens");

# Test 8: Embeddings
my $emb = $cache->embeddings(0, "The quick brown fox");
ok(ref $emb eq 'ARRAY', "embeddings returns arrayref");
ok(scalar @$emb > 0, "embeddings has elements");
ok(scalar @$emb == $n_embd, "embeddings length matches n_embd ($n_embd)");

# Test 9: Stats
my $stats = $cache->get_stats();
ok(ref $stats eq 'HASH', "get_stats returns hashref");
ok($stats->{tokens_total} > 0, "stats tokens_total > 0");
ok($stats->{prompt_tokens_total} > 0, "stats prompt_tokens_total > 0");
ok($stats->{completion_tokens_total} > 0, "stats completion_tokens_total > 0");

# Test 10: Free slot
$cache->free_slot(0);
my $slots_after = $cache->get_slots();
ok($slots_after->{0}{state} eq 'idle', "slot 0 is idle after free");

# Test 11: KV cache save/load
my $cache_file = "$cache->{cache_dir}/test.cache";

my $saved_bytes = $cache->save_slot_cache(1, $cache_file);
ok($saved_bytes > 0, "save_slot_cache saved $saved_bytes bytes");
ok(-f $cache_file, "cache file exists");

my $loaded_bytes = $cache->load_slot_cache(1, $cache_file);
ok($loaded_bytes > 0, "load_slot_cache loaded $loaded_bytes bytes");

# Test 12: List cached slots
my @cached = $cache->list_cached_slots();
ok(scalar @cached > 0, "list_cached_slots found files");

# Test 13: Reset stats
$cache->reset_stats();
my $stats_after = $cache->get_stats();
ok($stats_after->{tokens_total} == 0, "stats reset to 0");

$cache->DESTROY;
unlink $cache_file if -f $cache_file;
rmdir $cache->{cache_dir} if -d $cache->{cache_dir};

done_testing();
