use strict; use warnings;

use Test::More tests => 13;

use FindBin;
use lib "$FindBin::Bin/..";
use lib "$FindBin::Bin/../blib/arch";

use_ok("Llama");

# Test 2: Backend init/free
eval {
    Llama::backend_init();
    Llama::backend_free();
};
is($@, "", 'backend init/free works');

# Test 3: Model can be loaded
my $model_path = $ENV{GGUF_MODEL} // 'Qwen3.5-4B-ROCMFP4.gguf';
my $model;
eval {
    Llama::set_log_callback(sub {print $_[0]});
    Llama::backend_init();
    $model = Llama::model_load($model_path);
};
is($@, "", "no die for init and model_load($model_path)");
ok(defined $model, "model $model_path loaded");
SKIP: {
skip "model $model_path failed to load", 9 unless defined $model;
ok(1, "model loaded successfully for $model_path: $model");

# Test 4: Model query functions
ok($model->n_ctx_train > 0, 'n_ctx_train returns positive value');
ok($model->n_embd > 0, 'n_embd returns positive value');
ok($model->n_layer > 0, 'n_layer returns positive value');
ok($model->n_params > 0, 'n_params returns positive value');
my $desc = $model->desc;
ok(length($desc) > 0, "model desc: $desc");

# Test 5: Context creation
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
} else {
    ok(1, 'context created successfully');
    ok($ctx->n_ctx == 2048, 'n_ctx matches requested');
}

# Test 6: Batch creation
my $batch;
eval {
    $batch = Llama::Batch->new(max_tokens => 64);
};
ok(!$@, 'batch created successfully');

# Cleanup
$batch->DESTROY if $batch;
$ctx->DESTROY if $ctx;
$model->DESTROY;

}
Llama::backend_free();
done_testing();
