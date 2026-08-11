use strict;
use warnings;
use ExtUtils::MakeMaker;

my $rocm_path = '/opt/rocm';
# Path to minijinja-cabi library (adjust as needed)
my $minijinja_src_dir = '../minijinja';  # Relative path from Minijinja/ dir  
my $lib_dir = "$minijinja_src_dir/target/release";
my $inc_dir = "$minijinja_src_dir/minijinja-cabi/include";

WriteMakefile(
    NAME         => 'Minijinja',
    VERSION_FROM => 'Minijinja.pm',
    PREREQ_PM    => {},
    dist         => { COMPRESS => 'gzip -9f', SUFFIX => 'gz' },
    
    OBJECT       => '$(O_FILES)',
    CC           => 'g++',
    LD           => 'g++',
    OPTIMIZE     => '-O2',
    
    LIBS         => [
        "-L$lib_dir", 
        "-Wl,-rpath,$lib_dir",
        "-lminijinja_cabi"
    ],
    
    INC          => "-I$inc_dir " .
                    "-DMINIJINJA_INCLUDE_DIR=\\\"$inc_dir\\\"",
                    
    clean        => { FILES => 'Minijinja.so Minijinja.bs' }
);
