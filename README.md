# Minijinja - Perl XS Bindings to minijinja-cabi

Perl bindings to [minijinja](https://github.com/mitsuhiko/minijinja), 
a fast Rust implementation of the Jinja2 template engine.

This module provides direct FFI bindings to the C ABI library built from minijinja-cabi.

## Requirements

- Perl 5.14+
- A working Rust toolchain (for building dependencies)  
- The minijinja repository cloned as a sibling directory:
  ```
  ├── llama.cpp.git/           # Parent repo containing this module
  │   └── minijinja/           # Required: git clone https://github.com/mitsuhiko/minijinja.git
  └── Minijinja/               # This CPAN distribution
      ├── lib/Minijinja.pm     # Main Perl module
      ├── Minijinja.xs         # XS binding code  
      ├── Makefile.PL          # Build configuration
      └── t/*.t                # Test suite
  ```

## Building

```bash
cd Minijinja
perl Makefile.PL
make
make test
make install
```

The build will look for `libminijinja_cabi.so` in `../minijinja/target/release`.

Make sure that path is included in your LD_LIBRARY_PATH at runtime, or copy/link 
the .so file into /usr/local/lib or similar.

## Basic Usage

```perl
use Minijinja;

# Create environment and render inline template  
my $result = Minijinja->apply_from_string(
    'Hello {{ name }}!',
    { name => 'World' }
);

# Or use object-oriented interface  
my $env = Minijinja->new();
$env->add_template('greeting', 'Welcome to {{ site_name }}!');
print $env->render('greeting', { site_name => 'MySite' });
```

## API Reference

### Constructor Methods

#### new() -> Minijinja

Create a new MiniJinja environment instance. Returns an opaque handle wrapped 
in the Minijinja package. Throws on failure.

### Instance Methods  

#### add_template($name, $source) -> bool

Register a template by name with its Jinja2 source code. Returns true on success.

#### render($template_name, \%context) -> string

Render a registered template with the given context hash reference. Returns the 
rendered output as a Perl string. Context values are limited to simple scalars:
strings, integers, floats (v1 limitation).

#### DESTROY()

Cleanup handler that frees the underlying C environment when the Perl object goes out of scope.

### Class Methods

#### apply_from_string($class_or_self, $template_src, \%context) -> string

One-shot rendering without explicit template registration. Useful for quick renders or testing.
If called as an instance method ($obj->apply_from_string(...)), reuses existing env. If called 
as class method (Minijinja->apply_from_string(...)), creates temporary env and destroys it after use.

### Error Handling

The XS layer provides error introspection functions:

- `error_exists()` - IV returning 0/1 indicating if last operation set an error state  
- `last_error_msg()` - char* pointer to human-readable error message from minijinja-cabi
- `cstring_free($ptr)` - Free memory allocated by minijinja-cabi for strings like error messages

## License

Apache 2.0 (same license as upstream minijinja project)

See LICENSE file in parent repository details.
