use strict;
use warnings;
use ExtUtils::MakeMaker;
use Config;

my $rocm_path = $ENV{ROCM_PATH} || '/opt/rocm';
my $llama_src = $ENV{LLAMA_SRC}
    or die "Set LLAMA_SRC to the llama.cpp source directory\n";
my $llama_build = $ENV{LLAMA_BUILD} || "$llama_src/build";

my $cc = $ENV{CXX} || 'g++';

WriteMakefile(
    NAME         => 'Llama',
    VERSION_FROM => 'Llama.pm',
    dist         => { COMPRESS => 'gzip -9f', SUFFIX => 'gz', },
    OBJECT       => '$(O_FILES)',
    CC           => $cc,
    LD           => $cc,
    OPTIMIZE     => '-O2 -std=c++17',
    LIBS         => [
        "-L$llama_build/bin -lllama -lggml-hip -lggml-base",
        "-L$rocm_path/lib -lrocblas -lhipblas -lamdhip64",
    ],
    INC          => "-I$llama_src/include -I$llama_src/ggml/include -I$rocm_path/include",
    LDDLFLAGS    => "-shared -Wl,--disable-new-dtags",
    CCFLAGS      => "$Config{ccflags} -Wextra -Werror -fPIC -rdynamic",
    DEFINE       => '-UXS_APIVERSION_BOOTCHECK -UXS_VERSION_BOOTCHECK',
    clean        => { FILES => 'Llama.so Llama.bs' },
    NO_META      => 1,
    PL_FILES     => {},
    NO_MYMETA    => 1,
    NO_PERLLOCAL => 1,
    NO_PACKLIST  => 1,
    NEEDS_LINKING=> 1,
    XSPROTOARG   => '-prototypes',
);
