use strict;
use warnings;

use Test::More;
use File::Temp qw(tempdir);

BEGIN {
    eval { require Llama::Cache; };
    if ($@) {
        plan skip_all => "Llama::Cache not loadable: $@";
        exit 0;
    }
}

# Test 1: Module loads
ok(1, 'Llama::Cache module loaded');

# Test 2: Create cache with model
my $model_path = $ENV{GGUF_MODEL} // 'Qwen3.5-4B-ROCMFP4.gguf';
my $cache;
eval {
    $cache = Llama::Cache->new(
        model_path => $model_path,
        n_ctx      => 2048,
        n_batch    => 256,
        n_threads  => 4,
        n_slots    => 4,
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

my $slot_2 = $cache->alloc_slot();
ok(defined $slot_2, "alloc_slot returned $slot_2");

my $slot_3 = $cache->alloc_slot();
ok(defined $slot_3, "alloc_slot returned $slot_3");

my $slot_busy = $cache->alloc_slot();
ok(!defined $slot_busy, "alloc_slot returns undef when all slots busy");

my $slots = $cache->get_slots();
ok(ref $slots eq 'HASH', "get_slots returns hashref");
ok(defined $slots->{0}, "slot 0 exists in slots");
ok(defined $slots->{1}, "slot 1 exists in slots");
ok(defined $slots->{2}, "slot 2 exists in slots");
ok(defined $slots->{3}, "slot 3 exists in slots");

# Test 5: Chat completion (blocking)
my $slot_id = 0;
my $result = $cache->chat_completion($slot_id, [{role => "user", content => "The quick brown fox"}], 16);
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

# Test 14: Auto-save on free_slot
my $slot_a = $cache->alloc_slot();
$cache->chat_completion($slot_a,[{role => "user", content => "Hello world test"}], 8);
my $n_tokens_before = $cache->get_slot($slot_a)->{n_tokens};
ok($n_tokens_before > 0, "slot $slot_a has $n_tokens_before tokens before free");

my $cache_file_2 = $cache->_slot_cache_path($slot_a);
$cache->free_slot($slot_a);
ok(-f $cache_file_2, "auto-save created cache file at $cache_file_2");
ok(-s $cache_file_2 > 0, "auto-save cache file has content");

# Test 15: Auto-restore on alloc_slot (slot_a is now free)
my $slot_b = $cache->alloc_slot();
my $n_tokens_after = $cache->get_slot($slot_b)->{n_tokens};
ok($n_tokens_after > 0 && $n_tokens_after < 1000, "auto-restore loaded $n_tokens_after tokens into slot $slot_b");

# Test 16: save_slot_to_mmap_file and load_slot_from_mmap_file
my $slot_c = $cache->alloc_slot();
my $messages2 = [{ role => "user", content => "Test mmap save" }];
$cache->chat_completion($slot_c, $messages2, 8);
my $mmap_cache_file = "$cache->{cache_dir}/mmap_test.cache";
my $saved = $cache->save_slot_to_mmap_file($slot_c, $mmap_cache_file);
ok($saved > 0, "save_slot_to_mmap_file saved $saved bytes");
ok(-f $mmap_cache_file, "mmap cache file exists");

my $slot_d = $cache->alloc_slot();
my $loaded = $cache->load_slot_from_mmap_file($slot_d, $mmap_cache_file);
ok($loaded > 0, "load_slot_from_mmap_file loaded $loaded bytes");
my $n_tokens_loaded = $cache->get_slot($slot_d)->{n_tokens};
ok($n_tokens_loaded > 0 && $n_tokens_loaded < 1000, "loaded slot has $n_tokens_loaded tokens");

# Test 17: conv_id slot reuse
my $conv_id = "test-conv-123";
my $slot_e = $cache->alloc_slot($conv_id);
ok(defined $slot_e, "alloc_slot with conv_id returned $slot_e");
my $messages3 = [{ role => "user", content => "First turn" }];
$cache->chat_completion($slot_e, $messages3, 8, { conv_id => $conv_id });
my $n_tokens_first = $cache->get_slot($slot_e)->{n_tokens};
ok($n_tokens_first > 0, "first turn has $n_tokens_first tokens");

my $slot_f = $cache->alloc_slot($conv_id);
is($slot_f, $slot_e, "alloc_slot with same conv_id returns same slot ($slot_f == $slot_e)");

my $slot_g = $cache->alloc_slot("different-conv");
ok(defined $slot_g && $slot_g != $slot_e, "alloc_slot with different conv_id returns different slot ($slot_g != $slot_e)");

# Test 18: mmap file I/O with offset (skip header)
my $slot_h = $cache->alloc_slot();
my $messages4 = [{ role => "user", content => "Test mmap offset" }];
$cache->chat_completion($slot_h, $messages4, 8);
my $mmap_offset_file = "$cache->{cache_dir}/mmap_offset_test.cache";
my $saved_offset = $cache->save_slot_to_mmap_file($slot_h, $mmap_offset_file);
ok($saved_offset > 0, "save_slot_to_mmap_file with offset saved $saved_offset bytes");

my $slot_i = $cache->alloc_slot();
my $loaded_offset = $cache->load_slot_from_mmap_file($slot_i, $mmap_offset_file);
ok($loaded_offset > 0, "load_slot_from_mmap_file with offset loaded $loaded_offset bytes");
my $n_tokens_loaded_offset = $cache->get_slot($slot_i)->{n_tokens};
ok($n_tokens_loaded_offset > 0 && $n_tokens_loaded_offset < 1000, "loaded slot with offset has $n_tokens_loaded_offset tokens");

# Test 19: auto-restore uses mmap when available
my $slot_j = $cache->alloc_slot();
my $messages5 = [{ role => "user", content => "Test auto-restore mmap" }];
$cache->chat_completion($slot_j, $messages5, 8);
my $n_tokens_before_free = $cache->get_slot($slot_j)->{n_tokens};
ok($n_tokens_before_free > 0, "slot $slot_j has $n_tokens_before_free tokens before free");
$cache->free_slot($slot_j);
my $slot_k = $cache->alloc_slot();
my $n_tokens_after_restore = $cache->get_slot($slot_k)->{n_tokens};
ok($n_tokens_after_restore > 0 && $n_tokens_after_restore < 1000, "auto-restore with mmap loaded $n_tokens_after_restore tokens into slot $slot_k");

# Cleanup
$cache->free_slot(0);
$cache->free_slot(1);
$cache->free_slot(2);
$cache->free_slot(3);
$cache->free_slot($slot_a) if defined $slot_a;
$cache->free_slot($slot_b) if defined $slot_b;
$cache->free_slot($slot_c) if defined $slot_c;
$cache->free_slot($slot_d) if defined $slot_d;
$cache->free_slot($slot_e) if defined $slot_e;
$cache->free_slot($slot_f) if defined $slot_f;
$cache->free_slot($slot_g) if defined $slot_g;
$cache->free_slot($slot_h) if defined $slot_h;
$cache->free_slot($slot_i) if defined $slot_i;
$cache->free_slot($slot_j) if defined $slot_j;
$cache->free_slot($slot_k) if defined $slot_k;
$cache->DESTROY;
unlink $cache_file if -f $cache_file;
unlink $mmap_cache_file if -f $mmap_cache_file;
unlink $mmap_offset_file if -f $mmap_offset_file;
rmdir $cache->{cache_dir} if -d $cache->{cache_dir};

done_testing();
