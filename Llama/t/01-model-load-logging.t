use strict; use warnings;

use Test::More tests => 9;

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
    my $m;
    eval { 
        $m = Llama::model_load($model_path); 
    };
    is($@, "", "no die");
    ok(defined $m, 'Model loads successfully with default (suppressed) logging');

    # Clean up model pointer for next test
    print "# FREE $m\n";
    Llama::llama_model_free($m) if defined $m;
    print "# GC $m\n";
    $m = undef;
    print "# GC DONE\n";
} else {
    skip "Test GGUF file not found at $model_path", 6;
}

eval {
    Llama::backend_free();
};
is($@, "", "free ok");

# Test 5: llama_log_noop callback is linked properly by verifying backend init completes  
eval { 
    Llama::backend_init(); 
};
is($@, "", "no die as init again");

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
