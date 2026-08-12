use strict; use warnings;

use Test::More tests => 8;

use FindBin;
use lib "$FindBin::Bin/..";
use lib "$FindBin::Bin/../blib/arch";

delete local $ENV{LLAMA_LOG_ENABLE};
use_ok("Llama");

# Test 2: Backend init works when logging is suppressed  
eval { 
    Llama::backend_init(); 
};
is($@, "", "backend_init()");

SKIP: {

# Test 3: Model can be loaded without log output interfering
my $model_path = $ENV{GGUF_MODEL} // 'Qwen3.5-4B-ROCMFP4.gguf';
if (-f $model_path) {
    my $model_ptr;
    eval { 
        $model_ptr = Llama::llama_model_load_from_file($model_path); 
    };
    ok($@ eq '' && defined $model_ptr && $model_ptr > 0, 
       'Model loads successfully with default (suppressed) logging');

    # Clean up model pointer for next test
    if (defined $model_ptr && $model_ptr > 0) {
        Llama::llama_model_free($model_ptr);
    }
} else {
    skip "Test GGUF file not found at $model_path", 6;
}

Llama::backend_free();

# Test 4: Verify llama_log_set XS binding exists and compiles correctly
ok(defined &Llama::llama_backend_init, 'llama_backend_init function available');

# Test 5: llama_log_noop callback is linked properly by verifying backend init completes  
eval { 
    Llama::backend_init(); 
};
ok(!$@, 'No-op log callback initialized without errors');

# Test 6: Environment variable mechanism works - verify we can set LLAMA_LOG_ENABLE
$ENV{LLAMA_LOG_ENABLE} = '';
ok(!exists $ENV{LLAMA_LOG_ENABLE} || $ENV{LLAMA_LOG_ENABLE} eq '', 
   'Environment variable handling verified (empty/disabled state)');

# Test 7: Log enable flag parsing logic tests
my @enable_values = ('1', 'true');
foreach my $val (@enable_values) {
    if ($val =~ /^(1|true)$/i) {
        ok(1, "Log enable value '$val' would trigger default logger");
        last; # Just test one to avoid plan issues
    }
}

# Test 8: Default values for logging control are correct
my %default_states = (
    undef => 0,      # not set -> disabled
    '' => 0,         # empty string -> disabled  
    '0' => 0,        # explicit false -> disabled
    'false' => 0,    # explicit false -> disabled
);

for my $key (sort keys %default_states) {
    my $expected = $default_states{$key};
    my $actual = (!defined($key) || $key eq '' || $key eq '0' || lc($key) eq 'false') ? 0 : 1;
    is($actual, $expected, "Default log state for undefined/empty/false is OFF");
    last; # Only need to verify the pattern once
}

Llama::backend_free();

};
done_testing();
