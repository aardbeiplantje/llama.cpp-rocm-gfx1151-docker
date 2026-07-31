use strict;
use warnings;

$ENV{LD_LIBRARY_PATH} = '/opt/rocm/lib' unless $ENV{LD_LIBRARY_PATH};

use Test::More;
use File::Temp qw(tempdir);
use HTTP::Tiny;
use JSON::PP;
use Fcntl qw(:flock);

BEGIN {
    $ENV{PERL5LIB} = '/workdir/llama.cpp.git/Llama/blib/lib:/workdir/llama.cpp.git/Llama/blib/arch' unless $ENV{PERL5LIB};
    unshift @INC, '/workdir/llama.cpp.git/Llama/blib/lib', '/workdir/llama.cpp.git/Llama/blib/arch';
    eval { require Llama::Cache };
    if ($@) {
        plan skip_all => "Llama::Cache not loadable: $@";
        exit 0;
    }
}

my $model_path = $ENV{GGUF_MODEL} // '/workdir/llama.cpp.git/llama.cpp/build/bin/Qwen3.5-4B-ROCMFP4.gguf';
my $cache_dir = tempdir('integration_cache_XXXXXX', CLEANUP => 0);
my $nginx_conf = "$cache_dir/nginx.conf";
my $nginx_pid_file = "$cache_dir/nginx.pid";
my $socket_path = "$cache_dir/llama.sock";

my $system_message = join(' ', map {
    my @words = ('the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog', 'and', 'then',
                 'continues', 'with', 'more', 'interesting', 'sentences', 'that', 'fill', 'up', 'the',
                 'context', 'window', 'with', 'meaningful', 'content', 'for', 'testing', 'purposes',
                 'throughout', 'the', 'entire', 'message', 'block', 'which', 'should', 'be', 'long',
                 'enough', 'to', 'demonstrate', 'how', 'system', 'messages', 'consume', 'tokens',
                 'and', 'affect', 'performance', 'measurements', 'in', 'real', 'world', 'scenarios');
    my $count = int(rand(15)) + 10;
    join(' ', @words[0 .. $count > scalar @words ? $#words : $count - 1]);
} 1 .. 100);

my $full_system = <<SYSMSG;
You are a helpful assistant. $system_message
SYSMSG

my $nginx_pid;
my $http = HTTP::Tiny->new(timeout => 30);

sub start_nginx {
    my $conf = <<CONF;
worker_processes 1;
error_log $cache_dir/nginx_error.log;
pid $nginx_pid_file;

load_module modules/ndk_http_module.so;
load_module modules/ngx_http_lua_module.so;
load_module modules/ngx_http_perl_module.so;

include /usr/lib/nginx/modules/*.conf;

events {
    worker_connections 16;
}

http {
    default_type application/json;
    access_log $cache_dir/nginx_access.log;
    error_log $cache_dir/nginx_error.log;

    perl_modules $ENV{PERL5LIB};
    perl_require Llama.pm;

    server {
        listen 18000;
        client_max_body_size 8m;

        location /api/cache/health {
            perl Llama::cache_health;
        }

        location /api/cache/v1/health {
            perl Llama::cache_health;
        }

        location /api/cache/v1/models {
            perl Llama::cache_models;
        }

        location /api/cache/v1/slots {
            perl Llama::cache_slots;
        }

        location /api/cache/v1/chat/completions {
            perl Llama::cache_chat;
        }

        location /api/cache/v1/completions {
            perl Llama::cache_completion;
        }

        location /api/cache/v1/embeddings {
            perl Llama::cache_embeddings;
        }

        location /api/cache/v1/tokenize {
            perl Llama::cache_tokenize;
        }

        location /api/cache/v1/detokenize {
            perl Llama::cache_detokenize;
        }

        location /api/cache/metrics {
            perl Llama::cache_metrics;
        }

        location /api/cache/props {
            perl Llama::cache_props;
        }

        location /api/cache/v1/chat/completions/input_tokens {
            perl Llama::cache_input_tokens;
        }

        location /api/cache/v1/stream {
            perl Llama::cache_stream;
        }

        location /api/cache/v1/streams/lookup {
            perl Llama::cache_streams_lookup;
        }

        location /api/cache/v1/stream/ {
            perl Llama::cache_stream_delete;
        }

        location /api/cache {
            perl Llama::cache_chat;
        }
    }
}
CONF

    open my $cfh, '>', $nginx_conf or die "Cannot write nginx conf: $!";
    print $cfh $conf;
    close $cfh;

    $nginx_pid = fork();
    die "Cannot fork: $!" unless defined $nginx_pid;

    if ($nginx_pid == 0) {
        $ENV{MODEL} = $model_path;
        $ENV{LD_LIBRARY_PATH} = '/opt/rocm/lib';
        exec('nginx', '-c', $nginx_conf, '-g', 'daemon off;') or die "Cannot exec nginx: $!";
    }

    for my $i (1 .. 30) {
        sleep 1;
        my $resp = $http->get("http://localhost:18000/api/cache/health");
        return if $resp->{status} == 200;
    }
    die "nginx did not start within 30 seconds";
}

sub stop_nginx {
    return unless defined $nginx_pid;
    kill 'TERM', $nginx_pid;
    waitpid($nginx_pid, 0);
    $nginx_pid = undef;
}

sub json_post {
    my ($path, $body) = @_;
    my $json = JSON::PP->new->utf8->canonical;
    my $body_str = $json->encode($body);
    my $resp = $http->post("http://localhost:18000$path", {
        content => $body_str,
        headers => { 'Content-Type' => 'application/json' },
    });
    return $resp;
}

sub json_get {
    my ($path) = @_;
    my $resp = $http->get("http://localhost:18000$path");
    return $resp;
}

# Test 1: Start nginx
ok(start_nginx(), 'nginx started successfully');

# Test 2: Health check
my $resp = json_get('/api/cache/health');
ok($resp->{status} == 200, 'health endpoint returns 200');
my $health = decode_json($resp->{content});
is($health->{status}, 'ok', 'health status is ok');

# Test 3: Models endpoint
$resp = json_get('/api/cache/v1/models');
ok($resp->{status} == 200, 'models endpoint returns 200');
my $models = decode_json($resp->{content});
ok(scalar @$models > 0, 'models returns at least one model');
my $model_name = $models->[0]{id};
ok(length($model_name) > 0, "model name is '$model_name'");

# Test 4: Slots endpoint
$resp = json_get('/api/cache/v1/slots');
ok($resp->{status} == 200, 'slots endpoint returns 200');
my $slots = decode_json($resp->{content});
ok(ref $slots eq 'HASH', 'slots returns hashref');
ok(scalar keys %$slots > 0, 'slots has entries');

# Test 5: Props endpoint
$resp = json_get('/api/cache/props');
ok($resp->{status} == 200, 'props endpoint returns 200');
my $props = decode_json($resp->{content});
ok(scalar @$props > 0, 'props returns at least one model');

# Test 6: Chat completion with system message
my $messages = [
    { role => "system", content => $full_system },
    { role => "user", content => "What is 2+2?" },
];
$resp = json_post('/api/cache/v1/chat/completions', {
    messages => $messages,
    n_predict => 16,
    stream => 0,
});
ok($resp->{status} == 200, "chat completion returned 200 (status=$resp->{status})");
my $chat_result = decode_json($resp->{content});
ok(defined $chat_result->{choices}, 'chat result has choices');
ok(scalar @{$chat_result->{choices}} > 0, 'chat result has at least one choice');
ok(defined $chat_result->{choices}[0]{message}{content}, 'choice has message content');
ok(defined $chat_result->{conv_id}, 'chat result has conv_id');
my $conv_id = $chat_result->{conv_id};
ok(length($conv_id) > 0, "conv_id is '$conv_id'");
my $first_prompt_tokens = $chat_result->{usage}{prompt_tokens};
ok($first_prompt_tokens > 0, "first request used $first_prompt_tokens prompt tokens");

# Test 7: Second turn with same conv_id (KV cache reuse)
$messages = [
    { role => "system", content => $full_system },
    { role => "user", content => "What is 2+2?" },
];
$resp = json_post('/api/cache/v1/chat/completions', {
    messages => $messages,
    n_predict => 16,
    conv_id => $conv_id,
});
ok($resp->{status} == 200, "second turn returned 200 (status=$resp->{status})");
my $chat_result2 = decode_json($resp->{content});
ok(defined $chat_result2->{choices}, 'second turn has choices');
my $second_prompt_tokens = $chat_result2->{usage}{prompt_tokens};
ok($second_prompt_tokens > 0, "second request used $second_prompt_tokens prompt tokens");
ok($second_prompt_tokens == $first_prompt_tokens,
    "same conv_id reused slot ($second_prompt_tokens == $first_prompt_tokens)");

# Test 8: Slots show conv_id
$resp = json_get('/api/cache/v1/slots');
ok($resp->{status} == 200, 'slots endpoint still works');
$slots = decode_json($resp->{content});
my $slot_with_conv;
for my $sid (keys %$slots) {
    if (defined $slots->{$sid}{conv_id} && $slots->{$sid}{conv_id} eq $conv_id) {
        $slot_with_conv = $sid;
        last;
    }
}
ok(defined $slot_with_conv, "found slot $slot_with_conv with conv_id '$conv_id'");

# Test 9: Streams lookup
$resp = json_post('/api/cache/v1/streams/lookup', { conv_id => $conv_id });
ok($resp->{status} == 200, 'streams/lookup returned 200');
my $lookup = decode_json($resp->{content});
ok($lookup->{found}, 'lookup found the conv_id');
ok(defined $lookup->{slot_id}, 'lookup has slot_id');

# Test 10: Metrics with conv_id
$resp = json_get('/api/cache/metrics');
ok($resp->{status} == 200, 'metrics returned 200');
my $metrics = decode_json($resp->{content});
ok(scalar @{$metrics->{slots}} > 0, 'metrics has slot entries');
my $metric_slot;
for my $s (@{$metrics->{slots}}) {
    if (defined $s->{conv_id} && $s->{conv_id} eq $conv_id) {
        $metric_slot = $s;
        last;
    }
}
ok(defined $metric_slot, 'metrics shows slot with conv_id');

# Test 11: Free slot by conv_id
$resp = $http->delete("http://localhost:18000/api/cache/v1/stream/$conv_id");
ok($resp->{status} == 200, "DELETE /stream/$conv_id returned 200");
my $delete_result = decode_json($resp->{content});
is($delete_result->{status}, 'ok', 'delete returned ok status');

# Test 12: Verify slot is freed
$resp = json_get('/api/cache/v1/slots');
$slots = decode_json($resp->{content});
my $still_has_conv;
for my $sid (keys %$slots) {
    if (defined $slots->{$sid}{conv_id} && $slots->{$sid}{conv_id} eq $conv_id) {
        $still_has_conv = 1;
        last;
    }
}
ok(!defined $still_has_conv, "conv_id '$conv_id' no longer in slots after delete");

# Test 13: New inference after free (slot reuse)
$messages = [
    { role => "system", content => $full_system },
    { role => "user", content => "What is 3+3?" },
];
$resp = json_post('/api/cache/v1/chat/completions', {
    messages => $messages,
    n_predict => 16,
});
ok($resp->{status} == 200, "new inference after free returned 200");
my $chat_result3 = decode_json($resp->{content});
ok(defined $chat_result3->{choices}, 'new inference has choices');
ok(defined $chat_result3->{conv_id}, 'new inference has conv_id');
my $new_conv_id = $chat_result3->{conv_id};
ok($new_conv_id ne $conv_id, "new inference got different conv_id ($new_conv_id != $conv_id)");

# Test 14: Tokenize endpoint
$resp = json_post('/api/cache/v1/tokenize', {
    input => "Hello world test",
});
ok($resp->{status} == 200, 'tokenize returned 200');
my $tok_result = decode_json($resp->{content});
ok(scalar @{$tok_result->{tokens}} > 0, 'tokenize returned tokens');
ok($tok_result->{count} > 0, 'tokenize returned count');

# Test 15: Detokenize endpoint
$resp = json_post('/api/cache/v1/detokenize', {
    tokens => $tok_result->{tokens},
});
ok($resp->{status} == 200, 'detokenize returned 200');
my $detok_result = decode_json($resp->{content});
ok(defined $detok_result->{text}, 'detokenize returned text');

# Test 16: Input tokens endpoint
$resp = json_post('/api/cache/v1/chat/completions/input_tokens', {
    input => "Count these tokens please",
    conv_id => $new_conv_id,
});
ok($resp->{status} == 200, 'input_tokens returned 200');
my $input_tok = decode_json($resp->{content});
ok(scalar @{$input_tok->{tokens}} > 0, 'input_tokens returned tokens');
ok($input_tok->{count} > 0, 'input_tokens returned count');

# Cleanup
stop_nginx();
unlink $nginx_conf;
unlink "$cache_dir/nginx_error.log";
unlink "$cache_dir/nginx_access.log";
rmdir $cache_dir if -d $cache_dir;

done_testing();
