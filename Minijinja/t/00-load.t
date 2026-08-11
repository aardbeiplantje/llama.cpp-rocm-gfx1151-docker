use strict;
use warnings;
use Test::More tests => 2;

BEGIN {
    use_ok('Minijinja') or BAIL_OUT("Cannot load Minijinja");
}

# Check if we can instantiate environment  
my $env;
eval { 
    $env = Minijinja->new(); 
};

if (!$env) {
    ok(0, "Environment creation failed - libminijinja_cabi.so may be unavailable");
    diag("Note: This is expected if minijinja-cabi library not found in LD_LIBRARY_PATH");
} else {
    ok(1, 'Created MiniJinja environment instance');
    
    # Cleanup on destruction handled by DESTROY
    
done_testing();

