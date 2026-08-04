use strict;
use warnings;
use ExtUtils::MakeMaker;

# Note: This is a pure-Perl module using FFI::Platypus - no C/XS compilation needed!
# The libminijinja_cabi.so must be present on system LD_LIBRARY_PATH or 
# MINIJINJA_LIBDIR environment variable for runtime loading via FFI::Platypus.

my $minijinja_libdir = $ENV{MINIJINJA_LIBDIR} || '/opt/minijinja/lib';

WriteMakefile(
    NAME         => 'Minijinja',  
    VERSION_FROM => 'lib/Minijinja.pm',  # Extracts version from package file
    
    PREREQ_PM    => {
        'Carp'       => 0,
        'Exporter'   => 0, 
        'Scalar::Util'=> 0,
        'FFI::Platypus'=> '2.07',  # Minimum required version
    },
    
    dist         => { COMPRESS => 'gzip -9f', SUFFIX => 'gz', },
    
    clean        => { FILES => '*.so *.bs blib' },
);

