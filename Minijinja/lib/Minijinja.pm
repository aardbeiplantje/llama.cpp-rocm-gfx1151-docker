package Minijinja;

use strict;
use warnings;
use Carp qw(croak);

# ============================================================================  
# High-level OO interface
# ============================================================================

sub new {
    my ($class) = @_;
    my $env = _mj_env_new();
    croak("Failed to create MiniJinja environment") unless defined($env) && $env != 0;
    bless({ env => $env }, $class);
}

sub DESTROY {
    my ($self) = @_;
    if (defined($self->{env}) && $self->{env} != 0) {
        _mj_env_free($self->{env});
        undef $self->{env};
    }
}

# Add template to environment registry  
sub add_template {
    my ($self, $name, $source) = @_;
    return _mj_env_add_template($self->{env}, $name // '', $source // '');
}

# Render a registered template with hash context (flat key-value pairs only for v1)
sub render {
    my ($self, $template_name, $context_hr) = @_;
    
    # Build mj_value object from Perl hashref  
    my $ctx_val = _build_context_object($context_hr || {});
    
    # Call render function - returns C string pointer that must be freed  
    my $result_ptr = _mj_env_render_template($self->{env}, $template_name // '', $ctx_val);
    
    # Convert result and free allocated memory 
    my $perl_result = defined($result_ptr) ? result_ptr_to_string($result_ptr) : '';
    
    # Free the context value we built (if any was created)  
    if (defined($ctx_val) && $ctx_val != 0) {
        _mj_value_decref($ctx_val);
    }
    
    return $perl_result;
}

# One-shot: render inline template source without registering it first  
sub apply_from_string {
    my ($class_or_self, $tmpl_source, $context_hr) = @_;
    
    # Determine name based on caller type or use default  
    my $name = ref($_[0]) ? 'inline' : 'Minijinja::apply';
    
    # Build context value from hashref  
    my $ctx_val = _build_context_object($context_hr || {});
    
    # Render directly from string - returns C string pointer  
    my $result_ptr = undef;
    eval {
        $result_ptr = _mj_env_render_named_str_raw(undef, undef, $name, $tmpl_source || '', $ctx_val || 0);
    };
    
    # Handle errors gracefully  
    if (!defined($result_ptr)) {
        warn("Template rendering failed") unless $? == 0;
        if (defined($ctx_val) && $ctx_val != 0) {
            _mj_value_decref($ctx_val); 
        }
        return '';
    }
    
    # Convert result and free allocated memory  
    my $perl_result = result_ptr_to_string($result_ptr);
    
    # Free the context value we built (if any was created)  
    if (defined($ctx_val) && $ctx_val != 0) {
        _mj_value_decref($ctx_val);
    }
    
    return defined($perl_result) ? $perl_result : '';
}

# ============================================================================
# Helper functions to build complex values recursively  
# For now v1 only supports flat hashes with scalar values (strings/nums/bools)
# Future versions can support nested structures by extending this logic.  
# ============================================================================

sub _build_context_object {
    my ($hash_ref) = @_;
    
    # Start with empty object value  
    my $obj = _mj_value_new_object();
    
    for my $key (keys %{$hash_ref || {}}) {
        my $val = $hash_ref->{$key};
        
        # Build appropriate mj_value based on Perl type  
        my $vptr;
        if (!defined($val)) {
            $vptr = _mj_value_new_none();  
        } elsif ($val eq 'true') {
            $vptr = _mj_value_new_bool(1); 
        } elsif ($val eq 'false') {
            $vptr = _mj_value_new_bool(0);
        } elsif ($val =~ /^-?\d+\.?\d*$/) {  # Numeric check (int or float)  
            if ($val =~ /\./) {
                $vptr = _mj_value_new_f64($val + 0.0);
            } else {
                $vptr = _mj_value_new_i64(int($val));
            }
        } else {  
            # String value - must be string type to distinguish from numbers above  
            $vptr = _mj_value_new_string($val);
        } 
        
        # Set this key-value pair in the object (consumes both key and value!)
        _mj_value_set_string_key($obj, $key, defined($vptr) ? $vptr : _mj_value_new_none());
    } 
    
    return $obj;
}

# ============================================================================
# Utility functions exposed at package level  
# ============================================================================

sub result_ptr_to_string {
    my ($ptr) = @_;
    return '' unless defined($ptr) && $ptr != 0;
    
    use FFI::Platypus;
    my $ffi = FFI::Platypus->new(api => 1, lang => 'C'); 
    my $perl_str = $ffi->cast('string')($ptr);  # This copies the C string content
    
    # Free original allocation from Rust side  
    _mj_str_free($ptr);
    
   return defined($perl_str) ? $perl_str : '';
}


# ============================================================================
# Low-level FFI bindings to libminijinja_cabi.so  
# Using lazy initialization via BEGIN block below.  
# All functions prefixed with underscore (_) to indicate private usage.
# ============================================================================

my ($ffi_env_new, $ffi_env_free, $ffi_env_add_template, 
    $ffi_render_tmpl, $ffi_render_named_str, $ffi_val_none, 
    $ffi_val_undefined, $ffi_val_string, $ffi_val_bool,
    $ffi_val_u32, $ffi_val_i64, $ffi_val_f64, $ffi_obj_new,
    $ffi_key_setter, $ffi_decref, $ffi_str_free);

BEGIN {
    eval { require FFI::Platypus };
    if ($@) { die("Minijinja requires FFI::Platypus - install with: cpanm FFI::Platypus") }
    
    my $plat = FFI::Platypus->new(api => 1, lang => 'C');
    # Library name without prefix/suffix - system will find libminijinja_cabi.{so|dylib}
    $plat->lib('minijinja_cabi');
    
    # Environment lifecycle  
    ($ffi_env_new)        = $plat->attach(mj_env_new       => ['void*' => 'opaque']);
    ($ffi_env_free)       = $plat->attach([mj_env_free=>'void'], ['opaque', 'void*']);
    ($ffi_env_add_template)=$plat->attach([mj_env_add_template=>'bool'], 
                                           ['void*', 'opaque', 'string', 'string']);
    
    # Rendering functions - return allocated C string pointer that must be freed with str_free()
    ($ffi_render_tmpl)   = $plat->attach([mj_env_render_template=>'string_pointer'], 
                                          ['void*', 'opaque', 'string', 'opaque']);
    ($ffi_render_named_str)=$plat->attach([mj_env_render_named_str=>'string_pointer'], 
                                           ['void*', 'opaque', 'string', 'string', 'opaque']);
    
    # Value construction (all return opaque handles to internally managed values)  
    ($ffi_val_none)      = $plat->attach(mj_value_new_none     => ['void*' => 'opaque']);
    ($ffi_val_undefined) = $plat->attach(mj_value_new_undefined=>['void*' => 'opaque']);
    ($ffi_val_string)    = $plat->attach([mj_value_new_string  =>'opaque'], ['void*','string']);
    ($ffi_val_bool)      = $plat->attach([mj_value_new_bool    =>'opaque'], ['void*','int8_t']);
    ($ffi_val_u32)       = $plat->attach([mj_value_new_u32     =>'opaque'], ['void*','uint32_t']);
    ($ffi_val_i64)       = $plat->attach([mj_value_new_i64     =>'opaque'], ['void*','int64_t']);
    ($ffi_val_f64)       = $plat->attach([mj_value_new_f64     =>'opaque'], ['void*','f64']);
    ($ffi_obj_new)       = $plat->attach(mj_value_new_object   => ['void*' => 'opaque']);
    
    # Object mutation - takes mutable reference to object, sets key-value pair (consumes value!)  
    ($ffi_key_setter)    = $plat->attach([mj_value_set_string_key=>'bool'], 
                                          ['void*', 'opaque', 'string', 'opaque']);
    
    # Memory management for values and strings  
    ($ffi_decref)        = $plat->attach([mj_value_decref=>'void'], ['void*', 'opaque']);
    ($ffi_str_free)      = $plat->attach([mj_str_free=>'void'], ['void*', 'string']);
}

# ============================================================================
# Wrapper functions using FFI handles above (internal/private usage only)
# ============================================================================

sub _mj_env_new { return $ffi_env_new(undef) }
sub _mj_env_free { my($e)=@_; $ffi_env_free($e, undef); }

sub _mj_env_add_template {
    my($env, $name, $source)=@_;
    return 0 unless defined($env) && $env != 0;
    return $ffi_env_add_template(undef, $env, $name, $source);
}

sub _mj_env_render_template {
    my($env, $tmpl_name, $ctx_val)=@_;
    return undef unless defined($env) && $env != 0; 
    return $ffi_render_tmpl(undef, $env, $tmpl_name, $ctx_val || 0);
}

sub _mj_env_render_named_str_raw {
    my($scope,$n,$src,$val)=@_;  
    # Note: This is a raw wrapper - caller must handle context building and cleanup!
    return undef unless defined($src);
    return $ffi_render_named_str($scope//undef, undef//undef, $n//'inline', $src, $val||0);
}


# Value construction wrappers (internal/private)
sub _mj_value_new_none      { return $ffi_val_none(undef) }
sub _mj_value_new_undefined { return $ffi_val_undefined(undef) }  
sub _mj_value_new_string { my($s)=shift; return $ffi_val_string(undef,$s//='') }
sub _mj_value_new_bool     { my($b)=shift; return $ffi_val_bool(undef,$b?1:0) }
sub _mj_value_new_u32      { my($v)=shift; return $ffi_val_u32(undef,int($v)) }
sub _mj_value_new_i64      { my($v)=shift; return $ffi_val_i64(undef,int($v)) }
sub _mj_value_new_f64      { my($v)=shift; return $ffi_val_f64(undef,float($v)) }

sub _mj_value_new_object   { return $ffi_obj_new(undef) }

# Set key-value pair in object - consumes the value handle! Returns bool success. 
sub _mj_value_set_string_key {
    my ($obj_ptr, $key_str, $val_ptr) = @_;
    # Note: obj_ptr is passed by value but Rust internally gets &mut reference via ffi_fn macro magic
    return 0 unless defined($obj_ptr) && $obj_ptr != 0; 
    return $ffi_key_setter(undef, $obj_ptr, $key_str // '', $val_ptr || 0);  
}

# Memory management (internal/private)  
sub _mj_value_decref { my($p)=@_; $ffi_decref(undef,$p) if defined($p)&&$p!=0 }
sub _mj_str_free     { my($p)=@_; $ffi_str_free(undef,$p//='') if defined($p)&&$p!=0 }


1;

