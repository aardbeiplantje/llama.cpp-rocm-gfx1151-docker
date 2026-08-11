package Llama::Minijinja;

use strict;
use warnings;

our $VERSION = "0.1.0";

# High-level OO interface wrapping XS bindings to libminijinja_cabi.so

# Create new environment instance
sub new {
    my ($class) = @_;
    my $env_handle = mj_env_new();
    croak("Failed to create MiniJinja environment") unless defined($env_handle) && $env_handle != 0;
    bless({ _handle => $env_handle }, $class);
}

# Cleanup on destruction  
sub DESTROY {
    my ($self) = @_;
    if (defined($self->{_handle}) && $self->{_handle} != 0) {
        mj_env_free($self->{_handle});
        undef $self->{_handle};
    }
}

# Add template by name and source code  
sub add_template {
    my ($self, $name, $source) = @_;
    return 0 unless defined($self->{_handle}) && $self->{_handle} != 0;
    return mj_add_template($self->{_handle}, $name // "", $source // "");
}

# Render registered template with flat hash context (v1 only supports simple values)  
sub render {
    my ($self, $template_name, $context_hr) = @_;
    
    # XS function handles value conversion internally - for v1 we pass empty context ref  
    my $result_ptr = mj_render_simple(
        $self->{_handle} || 0, 
        $template_name // "inline", 
        \%{$context_hr || {}}
    );
    
    return defined($result_ptr) ? $result_ptr : "";
}

# One-shot inline template rendering without registration  
sub apply_from_string {
    my ($caller, $template_src, $context_hr) = @_;
    
    # Determine environment handle based on caller type  
    my $env_handle;
    if (ref($caller)) {
        # Called as method on object instance - use existing env  
        $env_handle = $caller->{_handle};
        
        # Perform rendering using existing env  
        return mj_render_simple($env_handle, undef, $template_src // "", \%{$context_hr || {}});
        
    } else {
        # Class method - create temp env and destroy after use  
        $env_handle = mj_env_new() or croak("Failed to create MiniJinja environment");
        
        # Perform rendering then cleanup immediately  
        my $result_ptr = mj_render_simple(
            $env_handle, 
            "inline", 
            $template_src // "", 
            \%{$context_hr || {}}
        );
        
        mj_env_free($env_handle);  # Destroy temporary environment
        
        return defined($result_ptr) ? $result_ptr : "";
    }
}


1;

__END__

=head1 NAME

Llama::Minijinja - Perl XS bindings to minijinja-cabi Rust library  

=cut
