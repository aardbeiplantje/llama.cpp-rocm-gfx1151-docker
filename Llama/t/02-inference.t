use strict; use warnings;

use Test::More tests => 14;

use FindBin;
use lib "$FindBin::Bin/..";
use lib "$FindBin::Bin/../blib/arch";

# Test 1: Module loads
use_ok("Llama");

# Test 2: Backend init
eval { Llama::backend_init(); };
is($@, "", 'backend init works');

# Test 3: Load model
my $model_path = $ENV{GGUF_MODEL} // 'Qwen3.5-4B-ROCMFP4.gguf';
my $model;
eval {
    $model = Llama::model_load($model_path);
};
is($@, "", "no die model_load($model_path)");
ok(defined $model, "model loaded");

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
is($@, "", "no die Llama::Context->new()");
ok(defined $ctx, 'context created successfully');

# Test 5: Get vocab
my $vocab = $model->vocab;
ok(defined $vocab, 'vocab retrieved from model');
my $nvocab = $vocab->n_tokens;
ok($nvocab > 0, "vocab has $nvocab tokens");

# Test 6: Tokenize a prompt
my $prompt = "The quick brown fox";
my @tokens = $vocab->tokenize($prompt);
ok(scalar @tokens > 0, "tokenized prompt into " . scalar(@tokens) . " tokens");

# Test 7: Decode the prompt batch
my $batch = Llama::Batch->new(max_tokens => scalar @tokens);
my @token_data = map { [$tokens[$_], $_, 0] } 0 .. $#tokens;
$batch->set_tokens(@token_data);
my $decode_ret = $ctx->decode($batch);
ok($decode_ret == 0, "decode returned status $decode_ret");

# Test 8: Get logits for last token
my $logit = $ctx->get_logits_ith(scalar(@tokens) - 1);
ok(defined $logit, "got logits for token " . (scalar(@tokens) - 1));

# Test 9: Token to piece
my $last_tok = $tokens[-1];
ok(defined $last_tok, "last TOK defined");
my $piece = $vocab->token_to_piece($last_tok);
ok(length($piece) > 0, "token_to_piece returned non-empty string: '$piece'");

# Test 10: Test greedy sampler
my $greedy = Llama::greedy_sampler();
ok(defined $greedy, 'greedy sampler created');
Llama::llama_sampler_free($greedy);

done_testing();
