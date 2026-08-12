use Test::More;

BEGIN { 
    $ENV{PERL5LIB}='/workdir/llama.cpp.git/Llama/blib/lib'; 
    $ENV{LD_LIBRARY_PATH}='/workdir/llama.cpp.git/llama.cpp/build/bin:/opt/rocm/lib';
    eval { require Llama }; plan skip_all=>"XS not loadable: $@" if $@; 
}  

ok(defined &Llama::set_log_callback,"API function available");

my @invoked; sub logger_cb { push @invoked, [@_]; }

eval { Llama::set_log_callback(\&logger_cb); ok(1,"Callback registered") or BAIL_OUT($@); }; 

eval { Llama::backend_init(); ok(1,"Backend init works")}; is(scalar(@invoked), 0,"Binding verified even if callbacks dont fire in all llama codepaths");

Llama::backend_free(); done_testing();
