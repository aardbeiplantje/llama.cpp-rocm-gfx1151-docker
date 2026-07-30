package llama;

use strict; use warnings;

use nginx;
use JSON::XS;
use POSIX qw(strftime);
use File::Path qw(make_path);

our $_json;
our $CACHE;

BEGIN {
    $ENV{LD_LIBRARY_PATH} = '/opt/rocm/lib:/llama/bin' unless $ENV{LD_LIBRARY_PATH};
    eval { require Llama::Cache };
    if ($@) {
        print STDERR "[llama.pm] WARNING: Llama::Cache not available: $@";
    } else {
        my $preset_file = $ENV{PRESET_FILE} || "/models/llamacpp_presets.ini";
        my $models_env = $ENV{MODELS};
        my @models;

        if ($models_env) {
            for my $model_spec (split /\|/, $models_env) {
                my ($path, $name) = split /:/, $model_spec, 2;
                next unless $path;
                push @models, {
                    path => $path,
                    name => $name || ($path =~ s/\.gguf$//r),
                    n_slots => 4,
                };
            }
        } elsif (my $model_path = $ENV{MODEL_PATH} || "/models/default.gguf") {
            push @models, {
                path => $model_path,
                name => ($model_path =~ s/\.gguf$//r),
                n_slots => 4,
            };
        }

        eval {
            if (@models) {
                $CACHE = Llama::Cache->new(
                    models    => \@models,
                    preset_file => $preset_file,
                    n_slots   => 4,
                    cache_dir => "/dev/shm/llama_cache",
                );
                print STDERR "[llama.pm] Llama::Cache initialized with " . scalar(@models) . " model(s)\n";
            } else {
                $CACHE = Llama::Cache->new(
                    model_path => $ENV{MODEL_PATH} || "/models/default.gguf",
                    n_ctx      => 4096,
                    n_batch    => 512,
                    n_threads  => 16,
                    n_slots    => 4,
                    cache_dir  => "/dev/shm/llama_cache",
                );
                print STDERR "[llama.pm] Llama::Cache initialized with model_path=$ENV{MODEL_PATH}\n";
            }
        };
        if ($@) {
            print STDERR "[llama.pm] ERROR initializing Llama::Cache: $@";
        }
    }
}

sub fixup {
    my ($r) = @_;
    my $r_ok = eval {
        $r->has_request_body(\&handle_req);
        return OK;
    };
    if($@){
        chomp(my $err = $@);
        print_error("[ERROR] request: $err");
        $r->send_http_header;
        $r->status(500);
        return OK;
    }
    return $r_ok;
}

sub handle_req {
    my ($r) = @_;
    my $method = $r->request_method;
    my $slot_id = $r->header_in("X-LLamaCPP-Id-slot");
    my $rb = $r->request_body();
    my $rf = $r->request_body_file();
    if(!defined $rb and defined $rf){
        open(my $b_fh, "<", $rf) or do {
            print_error("[ERROR] cant open tmp body file $rf: $!");
            return HTTP_INTERNAL_SERVER_ERROR;
        };
        local $/;
        $rb = <$b_fh>;
        close $b_fh;
    }
    $rb //= "{}";
    print_error("[INFO] REQUEST $method, header X-LLamaCPP-Id-slot: ".( $slot_id//"<no id_slot>"));
    if(length($slot_id) and $slot_id =~ m/^\d+$/){
        eval {
            #print_error("[debug] body: $rb, ".($rf//"<no file>"));
            my $req = JSON::XS::decode_json($rb);
            $_json //= JSON::XS->new->utf8->allow_blessed->allow_unknown->allow_nonref->convert_blessed->canonical;
            #print_error("[debug] orig: ",$_json->encode($req));
            print_error("[INFO] REQUEST $method, header X-LLamaCPP-Id-slot: ".( $slot_id//"<no id_slot>").", model:$req->{model}");

            do_magic_fixes($r, $req, $slot_id);

            $rb = $_json->encode($req);
            #print_error("[debug] new:  ",$rb);

            # log modified request body with 5-min rotation
            eval {
                my $log_dir = "/tmp/request-logs";
                make_path($log_dir) unless -d $log_dir;
                my $log_file = strftime("$log_dir/requests_%Y%m%d_%H%M00.log", localtime());
                open(my $fh, ">>", $log_file) or do {
                    print_error("[WARN] cannot open log file $log_file: $!");
                    return;
                };
                my $timestamp = localtime();
                my $model = $req->{model} // "unknown";
                print_error("[INFO] Logging request body for model=$model to $log_file");
                (my $safe_rb = $rb) =~ s/[\r\n]+//g;
                print $fh "[$timestamp] method=$method slot_id=" . ($slot_id // "none") . " model=$model $safe_rb\n";
                close $fh;

                # Update latest symlink to point to current 5-min file
                my $link = "$log_dir/latest";
                unlink $link if -e $link;
                symlink($log_file, $link) or print_error("[WARN] cannot create symlink $link: $!");
            };
            if($@){
                print_error("[WARN] failed to log request body: $@");
            }
        };
        if($@){
            chomp(my $err = $@);
            print_error("[error] $err");
        }
    }
    $r->send_http_header("application/json");
    $r->print($rb);
    return OK;
}

sub do_magic_fixes {
    my ($r, $llm_req, $slot_id) = @_;

    # add id_slot, make it a number
    $llm_req->{id_slot} = 0+$slot_id;

    # fix for qwen: first message ALWAYS a system, the rest NEVER a system
    $llm_req->{messages}[0]{role} = "system";
    $_->{role} = "user" for grep {$_->{role} eq "system"} ((@{$llm_req->{messages}})[1..$#{$llm_req->{messages}}]);

    # parse the first message's content (if role==system) and split it into
    # multiple in order to improve KV cache reuse. Note that the first message
    # will always have role==system at the moment, see the qwen hack
    my $m = \$llm_req->{messages}[0]{content};
    my $model_env;
    if($$m =~ s/^(You are powered by the model named .*?\. The exact model ID is .*?\n)//gms){
        $model_env = $1;
        # strip Today's date: from env
        $model_env =~ s/Today's date: .*?\n//ms;
    }
    my $project_env;
    if($$m =~ s/^(Here is some useful information about the environment you are running in:\n<env>.*?<\/env>)//gms){
        $project_env = $1;
    }
    my $mcp_instructions;
    if($$m =~ s/^(<mcp_instructions>.*?<\/mcp_instructions>)//gms){
        $mcp_instructions = $1;
    }
    my $skills;
    if($$m =~ s/^(Skills provide specialized instructions and workflows for specific tasks.*?<available_skills>.*?<\/available_skills>)//gms){
        $skills = $1;
    }
    my $agents_instructions;
    if($$m =~ s/^(Instructions from: .*?\/AGENTS\.md.*)//gms){
        $agents_instructions = $1;
    }

    my $fr = shift @{$llm_req->{messages}//[]};
    unshift @{$llm_req->{messages}},
        $fr,
        (length($mcp_instructions//"")?(
            {role => "user", content => $mcp_instructions},
            {role => "assistant", content => "Understood."}
        ):()),
        (length($skills//"")?(
            {role => "user", content => $skills},
            {role => "assistant", content => "Understood."},
        ):()),
        (length($model_env//"")?(
            {role => "user", content => $model_env},
            {role => "assistant", content => "Understood."},
        ):()),
        (length($agents_instructions//"")?(
            {role => "user", content => $agents_instructions},
            {role => "assistant", content => "Understood."},
        ):()),
        (length($project_env//"")?(
            {role => "user", content => $project_env},
            {role => "assistant", content => "Understood."},
        ):());

    return;
}

sub print_error {
    print STDERR @_,"\n";
}

# ============================================================================
# Cache API handlers — route to Llama::Cache instead of llama-server
# ============================================================================

sub cache_chat {
    my ($r) = @_;
    return HTTP_INTERNAL_SERVER_ERROR unless _check_cache($r);
    $r->send_http_header("application/json");
    my $rb = _read_body($r);
    my $req = JSON::XS::decode_json($rb);
    return _json_response($r, { error => "invalid JSON" }, 400) unless $req;

    my $slot_id = _get_slot_id($r, $req);
    my $messages = $req->{messages};
    my $n_predict = $req->{n_predict} // 256;
    my $stream = $req->{stream} // 0;

    if ($stream) {
        return _stream_chat($r, $slot_id, $messages, $n_predict, $req);
    }

    my $result;
    eval {
        $result = $CACHE->chat_completion($slot_id, $messages, $n_predict, $req);
    };
    if ($@) {
        print_error "[cache] chat_completion error: $@";
        return _json_response($r, { error => $@ }, 500);
    }

    return _json_response($r, $result);
}

sub cache_completion {
    my ($r) = @_;
    return HTTP_INTERNAL_SERVER_ERROR unless _check_cache($r);
    $r->send_http_header("application/json");
    my $rb = _read_body($r);
    my $req = JSON::XS::decode_json($rb);
    return _json_response($r, { error => "invalid JSON" }, 400) unless $req;

    my $slot_id = _get_slot_id($r, $req);
    my $prompt = $req->{prompt} // "";
    my $n_predict = $req->{n_predict} // 256;

    my $result;
    eval {
        $result = $CACHE->completion($slot_id, $prompt, $n_predict, $req);
    };
    if ($@) {
        print_error "[cache] completion error: $@";
        return _json_response($r, { error => $@ }, 500);
    }

    return _json_response($r, $result);
}

sub cache_embeddings {
    my ($r) = @_;
    return HTTP_INTERNAL_SERVER_ERROR unless _check_cache($r);
    $r->send_http_header("application/json");
    my $rb = _read_body($r);
    my $req = JSON::XS::decode_json($rb);
    return _json_response($r, { error => "invalid JSON" }, 400) unless $req;

    my $slot_id = _get_slot_id($r, $req);
    my $input = $req->{input} // "";

    my $emb;
    eval {
        $emb = $CACHE->embeddings($slot_id, $input);
    };
    if ($@) {
        print_error "[cache] embeddings error: $@";
        return _json_response($r, { error => $@ }, 500);
    }

    my $result = {
        object => "list",
        data => [{
            object => "embedding",
            index => 0,
            embedding => $emb,
        }],
        model => $CACHE->model_desc,
        usage => {
            prompt_tokens => scalar @{$CACHE->{contexts}[$slot_id]{n_tokens} // 0},
            total_tokens => scalar @{$CACHE->{contexts}[$slot_id]{n_tokens} // 0},
        },
    };

    return _json_response($r, $result);
}

sub cache_slots {
    my ($r) = @_;
    return HTTP_INTERNAL_SERVER_ERROR unless _check_cache($r);
    $r->send_http_header("application/json");

    my $method = $r->request_method;
    if ($method eq "GET") {
        my $slots = $CACHE->get_slots();
        return _json_response($r, $slots);
    } elsif ($method eq "POST") {
        my $rb = _read_body($r);
        my $req = JSON::XS::decode_json($rb);
        my $slot_id = $req->{id_slot};
        if (defined $slot_id) {
            $CACHE->free_slot($slot_id);
            return _json_response($r, { status => "ok", slot_id => $slot_id });
        }
        return _json_response($r, { error => "missing id_slot" }, 400);
    }

    return _json_response($r, { error => "method not allowed" }, 405);
}

sub cache_health {
    my ($r) = @_;
    $r->send_http_header("application/json");
    return _json_response($r, { status => "ok" });
}

sub cache_models {
    my ($r) = @_;
    return HTTP_INTERNAL_SERVER_ERROR unless _check_cache($r);
    $r->send_http_header("application/json");
    my $models = $CACHE->model_info();
    return _json_response($r, $models);
}

sub cache_tokenize {
    my ($r) = @_;
    return HTTP_INTERNAL_SERVER_ERROR unless _check_cache($r);
    $r->send_http_header("application/json");
    my $rb = _read_body($r);
    my $req = JSON::XS::decode_json($rb);
    return _json_response($r, { error => "invalid JSON" }, 400) unless $req;

    my $text = $req->{input} // "";
    my $vocab = $CACHE->{model}->vocab;
    my @tokens = $vocab->tokenize($text);

    return _json_response($r, { tokens => \@tokens, count => scalar @tokens });
}

sub cache_detokenize {
    my ($r) = @_;
    return HTTP_INTERNAL_SERVER_ERROR unless _check_cache($r);
    $r->send_http_header("application/json");
    my $rb = _read_body($r);
    my $req = JSON::XS::decode_json($rb);
    return _json_response($r, { error => "invalid JSON" }, 400) unless $req;

    my $tokens = $req->{tokens};
    return _json_response($r, { error => "tokens required" }, 400) unless $tokens;

    my $vocab = $CACHE->{model}->vocab;
    my @pieces;
    for my $tok (@$tokens) {
        push @pieces, $vocab->token_to_piece($tok);
    }
    my $text = join('', @pieces);

    return _json_response($r, { text => $text });
}

sub cache_metrics {
    my ($r) = @_;
    return HTTP_INTERNAL_SERVER_ERROR unless _check_cache($r);
    $r->send_http_header("application/json");

    my $slots = $CACHE->get_slots();
    my $stats = $CACHE->get_stats();

    my @slot_metrics;
    for my $id (sort keys %$slots) {
        my $s = $slots->{$id};
        push @slot_metrics, {
            id => $id,
            state => $s->{state},
            n_tokens => $s->{n_tokens},
            model => $s->{model} // "unknown",
        };
    }

    return _json_response($r, {
        slots => \@slot_metrics,
        stats => $stats,
    });
}

# ============================================================================
# Internal helpers
# ============================================================================

sub _check_cache {
    my ($r) = @_;
    if (!$CACHE) {
        $r->send_http_header("application/json");
        $r->print('{"error":"Llama::Cache not initialized"}');
        return 0;
    }
    return 1;
}

sub _read_body {
    my ($r) = @_;
    my $rb = $r->request_body();
    my $rf = $r->request_body_file();
    if (!defined $rb && defined $rf) {
        open(my $b_fh, "<", $rf) or do {
            print_error "[cache] cant open tmp body file $rf: $!";
            return "{}";
        };
        local $/;
        $rb = <$b_fh>;
        close $b_fh;
    }
    return $rb // "{}";
}

sub _get_slot_id {
    my ($r, $req) = @_;
    return $req->{id_slot} // 0 + 0;
}

sub _get_model {
    my ($r, $req) = @_;
    my $model_name = $req->{model};
    return $CACHE->get_model_by_name($model_name) if $model_name && $CACHE->get_model_by_name($model_name);
    return $CACHE->get_model();
}

sub _json_response {
    my ($r, $data, $status) = @_;
    $status //= 200;
    $_json //= JSON::XS->new->utf8->canonical->allow_blessed->allow_unknown->allow_nonref;
    $r->status($status) if $status != 200;
    $r->print($_json->encode($data));
    return OK;
}

sub _stream_chat {
    my ($r, $slot_id, $messages, $n_predict, $req) = @_;
    $r->send_http_header("text/event-stream");
    $r->connection("keep-alive");
    $r->buffered(0) if $r->can("buffered");

    my $conv_id = $req->{conv_id} // "perl-stream-" . time();
    my $stream = Llama::Cache::Stream->new($conv_id);

    my $slot = $CACHE->get_slot($slot_id);
    my $ctx = $slot->{context};
    my $vocab = $CACHE->{model}->vocab;

    my @tokens = @{$CACHE->_tokenize_messages($messages)};
    my $batch = Llama::Batch->new(max_tokens => scalar @tokens);
    $batch->set_tokens(map { [@tokens[$_], $_, 0] } 0 .. $#tokens);
    $ctx->decode($batch);
    $batch->DESTROY;
    $slot->{n_tokens} = scalar @tokens;
    $slot->{state} = "generating";

    my $n_ctx = $ctx->n_ctx;
    my $output = "";
    my $n_output = 0;

    for my $i (0 .. $n_predict - 1) {
        last if $slot->{n_tokens} >= $n_ctx;
        last if $stream->is_cancelled();

        my $last_tok = $tokens[-1];
        my $pos = $slot->{n_tokens} - 1;

        my $b = Llama::Batch->new(max_tokens => 1);
        $b->set_token(0, $last_tok, $pos, 0);
        $ctx->decode($b);
        $b->DESTROY;

        my $logit_ptr = Llama::_get_logits_ptr($ctx->{ptr});
        my @logits;
        for my $j (0 .. $vocab->n_tokens - 1) {
            $logits[$j] = Llama::_read_float($logit_ptr, $j);
        }
        my $max_logit = $logits[0];
        my $new_tok = 0;
        for my $j (1 .. $#logits) {
            if ($logits[$j] > $max_logit) {
                $max_logit = $logits[$j];
                $new_tok = $j;
            }
        }

        push @tokens, $new_tok;
        $slot->{n_tokens}++;
        $n_output++;

        my $piece = $vocab->token_to_piece($new_tok);
        $output .= $piece;
        $stream->add_chunk($piece);

        $r->print("data: " . $_json->encode({
            id      => $conv_id,
            object  => "chat.completion.chunk",
            created => time(),
            model   => $CACHE->model_desc,
            choices => [{
                index        => 0,
                delta        => { content => $piece },
                finish_reason => undef,
            }],
        }) . "\n\n");
        $r->rflush() if $r->can("rflush");
    }

    $slot->{state} = "idle";

    $r->print("data: " . $_json->encode({
        id      => $conv_id,
        object  => "chat.completion.chunk",
        created => time(),
        model   => $CACHE->model_desc,
        choices => [{
            index        => 0,
            delta        => {},
            finish_reason => "stop",
        }],
    }) . "\n\n");
    $r->print("data: [DONE]\n\n");
    return OK;
}

1;
