use strict;
use warnings;

use Test::More;

BEGIN {
    $ENV{PERL5LIB} = 'blib/lib:blib/arch' unless $ENV{PERL5LIB};
    eval { require Llama; };
    if ($@) {
        plan skip_all => "Llama XS module not loadable: $@";
        exit 0;
    }
}

# Test 1: Module loads
ok(1, 'Llama module loaded');

# Test 2: Backend init/free
eval { Llama::backend_init(); Llama::backend_free(); };
ok(!$@, 'backend init/free works');

# Test 3: Model can be loaded
my $model_path = $ENV{GGUF_MODEL} // '/workdir/llama.cpp.git/llama.cpp/build/bin/Qwen3.5-4B-ROCMFP4.gguf';
my $model;
eval {
    Llama::backend_init();
    $model = Llama::model_load($model_path);
};
if ($@) {
    ok(0, "model_load: $@");
    Llama::backend_free();
    plan skip_all => "Could not load test model";
    exit 0;
}
ok(1, 'model loaded successfully');

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
Llama::backend_free();

done_testing();
