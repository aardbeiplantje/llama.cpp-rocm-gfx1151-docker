use strict; use warnings;

use Test::More tests => 7;

use FindBin;
use lib "$FindBin::Bin/..";
use lib "$FindBin::Bin/../blib/arch";

use Data::Dumper;

use_ok("Llama");

ok(defined &Llama::set_log_callback,"API function available");

my @invoked;
sub logger_cb {
    push @invoked, @_;
}

eval {
    Llama::set_log_callback(\&logger_cb);
}; 
is($@, "", "no die set_log_callback(<sub>)");

eval {
    Llama::backend_init();
};
is($@, "", "no die backend_init()");

my $model_path = $ENV{GGUF_MODEL} // 'Qwen3.5-4B-ROCMFP4.gguf';
my $model = 
eval {
    Llama::model_load($model_path);
};
is($@, "", "no die model_load($model_path)");

eval {
    Llama::backend_free();
};
is($@, "", "no die backend_free()");
ok(scalar(@invoked) > 0, "log messages: >0 (545 currently)")
    or print Dumper(\@invoked);
done_testing();
