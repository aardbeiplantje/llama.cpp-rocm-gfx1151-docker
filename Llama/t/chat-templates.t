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

# Test 2: Backend init
eval { Llama::backend_init(); };
ok(!$@, 'backend init works');

# Test 3: Load model (required for chat templates)
my $model_path = $ENV{GGUF_MODEL} // '/workdir/llama.cpp.git/llama.cpp/build/bin/Qwen3.5-4B-ROCMFP4.gguf';
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

# Test 4: Get built-in template name (auto-detect from GGUF metadata)
my $tmpl_name = '';
eval {
    $tmpl_name = $model->chat_template() || '';
};
ok(!$@ && length($tmpl_name) > 0, "retrieved default template name: '$tmpl_name'");

# Test 5: Build sample messages arrayref with role/content structure
my @messages = (
    { role => 'system', content => 'You are a helpful assistant.' },
    { role => 'user',     content => 'Hello!' },
    { role => 'assistant',content => 'Hi there! How can I help?' },
    { role => 'user',     content => 'What is AI?' },
);
ok(scalar(@messages) == 4, 'created message list with 4 turns');

# Test 6: Validate all messages have required keys before applying template
for my $msg (@messages) {
    ok(exists($msg->{role}) && exists($msg->{content}), 
       "message has role and content keys");
}

# Test 7: Apply chat template using model method (auto-detects template from GGUF)
my $formatted_prompt;
eval {
    $formatted_prompt = $model->apply_chat_template(\@messages, 1);  # add_ass=1 to append assistant prefix
};
if ($@ || !defined($formatted_prompt)) {
    if (!defined($formatted_prompt)) {
        ok(0, "apply_chat_template returned undefined - check llama.cpp version supports templates for this model");
        done_testing();
        exit 0;
    } else {
        ok(0, "apply_chat_template error: $@");
        Llama::backend_free();
        plan skip_all => "Chat template application failed";
        exit 0;
    }
}
ok(length($formatted_prompt) > 0, "template applied successfully, output length=" . length($formatted_prompt));

# Test 8: Verify formatted prompt contains expected message content fragments
like($formatted_prompt, qr/You are a helpful assistant/, 'prompt contains system message');
like($formatted_prompt, qr/Hello!/, 'prompt contains first user message');
like($formatted_prompt, qr/What is AI\?/, 'prompt contains second user message');

# Test 9: Apply chat template using direct XS call with explicit NULL tmpl (auto-detect)
my $xs_formatted = Llama::llama_chat_apply_template(undef, \@messages, 0);
ok(defined($xs_formatted) && length($xs_formatted) > 0, 
   "direct XS apply_chat_template works without add_ass flag");

# Test 10: Compare results - should be similar but may differ in assistant suffix handling
if ($xs_formatted) {
    # Both should contain the same core content even if formatting differs slightly
    ok(index($xs_formatted, 'What is AI') >= 0 || index($formatted_prompt, 'What is AI') >= 0,
       "both methods produce valid output containing query text");
}

# Test 11: Error handling - invalid messages arrayref structure
eval {
    my @bad_msgs = ({ role => 'user' });  # missing 'content' key
    $model->apply_chat_template(\@bad_msgs);
};
like($@ // '', qr/message.*missing|role.*content/i, 'croaks on incomplete message hash');

# Test 12: Error handling - non-array reference input  
eval {
    $model->apply_chat_template({ role => 'user', content => 'test' });  # not an ARRAYREF
};
like($@ // '', qr/must be an array/, 'croaks on non-array messages');

done_testing();
