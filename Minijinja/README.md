# Minijinja Perl Module

Perl XS bindings to [minijinja](https://github.com/mitsuhiko/minijinja), 
a fast Rust implementation of the Jinja2 template engine.

## Building

Requires minijinja submodule cloned as sibling directory:

    cd ..
    git clone https://github.com/mitsuhiko/minijinja.git
    
Then build:

    perl Makefile.PL
    make
    sudo make install

## Usage (XS enabled)

```perl
use Minijinja;

my $env = Minijinja->new();
$env->add_template('hello', 'Hello {{ name }}!');
print $env->render('hello', {name => 'World'});  # "Hello World!"
```

## Files

- `Makefile.PL` - Build configuration  
- `Minijinja.pm` - Main Perl module with OO interface
- `typemap` - C type mappings for XS bridge  
- `t/*.t` - Test suite  

## License

Apache 2.0 (same as upstream minijinja project). See LICENSE file.
