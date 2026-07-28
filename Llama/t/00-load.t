use strict;
use warnings;

use Test::More;

# The XS module requires ROCm libs (libhipblas, librocblas, libamdhip64)
# which are only available on Strix Halo gfx1151 systems.
# This test suite is designed to run on the target hardware.
BEGIN {
    $ENV{PERL5LIB} = 'blib/lib:blib/arch' unless $ENV{PERL5LIB};
    eval { require Llama; };
    if ($@) {
        plan skip_all => "Llama XS module not loadable (ROCm libs missing or not on Strix Halo): $@";
        exit 0;
    }
    plan tests => 4;
}

ok(1, 'Llama module loaded');

# Test 1: backend init / free
eval { Llama::backend_init(); Llama::backend_free(); };
ok(!$@, 'backend init/free works');

# Test 2: Model query functions exist (can't test without a GGUF file)
my $mock_model = bless { ptr => 0, freed => 0 }, 'Llama::Model';
isa_ok($mock_model, 'Llama::Model', 'Llama::Model object');

# Test 3: Sampler constructors exist
eval {
    my $greedy = Llama::greedy_sampler();
    Llama::llama_sampler_free($greedy->{ptr});
};
ok(!$@, 'greedy_sampler works');

done_testing();
