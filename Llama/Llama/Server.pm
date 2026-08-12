package Llama::Server;

use strict; use warnings;

use Fcntl qw(:flock);

my $n_workers;
my $model_path;
my $model_ptr;
my @children;
my @data_rpipe;   # read end of data pipe from each child
my @data_wpipe;   # write end of data pipe to each child
my @ctrl_rpipe;   # read end of control pipe to each child
my @ctrl_wpipe;   # write end of control pipe to each child
my @ctx_ptrs;     # context pointer for each child

sub init {
    my (%opts) = @_;
    $n_workers         = $opts{n_workers}         || 4;
    $model_path        = $opts{model_path}        or die "model_path required";
    my $n_ctx          = $opts{n_ctx}             || 2048;
    my $n_batch        = $opts{n_batch}           || 512;
    my $n_threads      = $opts{n_threads}         || 16;
    my $n_threads_batch = $opts{n_threads_batch}  || 16;

    Llama::setup_rocm_env();
    Llama::backend_init();
    $model_ptr = Llama::llama_model_load_from_file($model_path);
    fork_workers($n_ctx, $n_batch, $n_threads, $n_threads_batch);
    return;
}

sub fork_workers {
    my ($n_ctx, $n_batch, $n_threads, $n_threads_batch) = @_;

    for my $i (0 .. $n_workers - 1) {
        my ($dr_r, $dr_w);  # data: child writes, parent reads
        my ($cr_r, $cr_w);  # control: parent writes, child reads

        pipe($dr_r, $dr_w) or die "pipe data: $!";
        pipe($cr_r, $cr_w) or die "pipe control: $!";

        my $pid = fork();
        die "fork: $!" unless defined $pid;

        if ($pid == 0) {
            # Child process
            close $dr_r;
            close $cr_w;

            # Each child creates its own context from the shared model
            my $ctx = Llama::llama_init_from_model(
                $model_ptr,
                $n_ctx, $n_batch, $n_threads, $n_threads_batch, 0
            );

            # Worker loop: read control commands, process, respond via data pipe
            while (my $cmd = <$cr_r>) {
                chomp $cmd;

                if ($cmd eq 'PING') {
                    print $dr_w "PONG\n";
                    next;
                }

                if ($cmd =~ /^DECODE\s+(\d+)\s+([\w,]+)/) {
                    my $n_predict = $1;
                    my $prompt_tokens_str = $2;

                    # Parse prompt tokens
                    my @prompt_tokens = split /,/, $prompt_tokens_str;

                    # Build batch from prompt
                    my $batch = Llama::llama_batch_init(scalar @prompt_tokens, 0, 1);
                    for my $j (0 .. $#prompt_tokens) {
                        Llama::llama_batch_set_token($batch, $prompt_tokens[$j], $j, [0], 1);
                    }

                    Llama::llama_decode($ctx, $batch);
                    Llama::llama_batch_free($batch);

                    # Generate tokens
                    my @output;
                    my $n_ctx_val = Llama::llama_n_ctx($ctx);
                    my $n_tokens = scalar @prompt_tokens;

                    my $sampler = _build_sampler();

                    for my $gen_i (0 .. $n_predict - 1) {
                        if ($n_tokens >= $n_ctx_val) {
                            last;
                        }

                        my $last_tok = $prompt_tokens[-1];
                        my $pos = $n_tokens - 1;

                        my $gen_batch = Llama::llama_batch_init(1, 0, 1);
                        Llama::llama_batch_set_token($gen_batch, $last_tok, $pos, [0], 1);
                        Llama::llama_decode($ctx, $gen_batch);
                        Llama::llama_batch_free($gen_batch);

                        my $new_tok = Llama::llama_sampler_sample($sampler, $ctx, 0);
                        push @prompt_tokens, $new_tok;
                        $n_tokens++;

                        my $piece = _token_to_piece($ctx, $new_tok);
                        push @output, $piece;
                    }

                    Llama::llama_sampler_free($sampler);

                    # Send result back
                    print $dr_w join('|', @output) . "\n";
                }
            }

            Llama::llama_free($ctx);
            Llama::llama_backend_free();
            exit 0;
        }

        # Parent process
        close $dr_w;
        close $cr_r;
        push @children, $pid;
        push @data_rpipe, $dr_r;
        push @ctrl_wpipe, $cr_w;
        # We don't know ctx ptr yet (it's in child), but we track via pipe
    }
}

sub _build_sampler {
    my $chain = Llama::llama_sampler_chain_init();
    Llama::llama_sampler_chain_add($chain, Llama::llama_sampler_init_top_k(64));
    Llama::llama_sampler_chain_add($chain, Llama::llama_sampler_init_top_p(0.8, 1));
    Llama::llama_sampler_chain_add($chain, Llama::llama_sampler_init_temp(0.8));
    return $chain;
}

sub _token_to_piece {
    my ($ctx, $token) = @_;
    my $buf = "\0" x 128;
    my $vocab_ptr = _get_vocab_from_ctx($ctx);
    my $len = Llama::llama_token_to_piece($vocab_ptr, $token, $buf, 128, 0, 0);
    if ($len < 0) {
        $len = -$len;
    }
    return substr($buf, 0, $len);
}

sub _get_vocab_from_ctx {
    my ($ctx) = @_;
    return Llama::llama_model_get_vocab(_get_model_from_ctx($ctx));
}

sub _get_model_from_ctx {
    my ($ctx) = @_;
    return Llama::llama_get_model($ctx);
}

sub decode {
    my ($self, $worker_id, $prompt, $n_predict) = @_;
    $n_predict //= 128;

    # Tokenize prompt in parent
    my @tokens = prompt_to_tokens($prompt);
    my $tok_str = join(',', @tokens);

    # Send decode request to worker
    my $wpipe = $ctrl_wpipe[$worker_id];
    print $wpipe "DECODE $n_predict $tok_str\n";
    flush $wpipe;

    # Read response
    my $rpipe = $data_rpipe[$worker_id];
    $rpipe->blocking(0);
    my $result;
    my $timeout = 60;  # seconds
    my $start = time();

    while (time() - $start < $timeout) {
        my $line = <$rpipe>;
        if (defined $line) {
            chomp $line;
            if ($line =~ /^(.+)$/) {
                $result = $1;
                last;
            }
        }
        sleep 0.01;
    }

    return $result // '[timeout]';
}

sub prompt_to_tokens {
    my ($text) = @_;
    # Simple char-to-token mapping for demo
    # In production, use Llama::Vocab->tokenize()
    return map { ord($_) } split //, $text;
}

sub workers {
    return @children;
}

sub worker_pid {
    my ($self, $i) = @_;
    return $children[$i];
}

sub DESTROY {
    my ($self) = @_;
    for my $i (0 .. $#children) {
        if (defined $children[$i]) {
            my $wpipe = $ctrl_wpipe[$i];
            print $wpipe "QUIT\n";
            close $wpipe;
        }
    }
    for my $pid (@children) {
        waitpid($pid, 0) if defined $pid;
    }
}

1;
