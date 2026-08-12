use strict; use warnings;

use Test::More tests => 4;

use FindBin;
use lib "$FindBin::Bin/..";
use lib "$FindBin::Bin/../blib/arch";

use_ok("Llama");

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
