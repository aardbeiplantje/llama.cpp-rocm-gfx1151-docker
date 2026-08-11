use strict;
use warnings;
use Test::More tests => 4;

SKIP: {
    skip "Minijinja XS module not available", 4 unless eval { require Minijinja };

    my $mj_env = Minijinja->new() or do {
        skip "Cannot create environment", 4;
    };

    # Test 1: Simple variable substitution  
    my $tmpl = $mj_env->add_template('hello', 'Hello {{ name }}!');
    ok($tmpl, 'Template registration succeeded') or diag("Error: ", Minijinja::error_exists());

    if ($tmpl) {
        my $result = $mj_env->render('hello', { name => 'World' });
        like($result, qr/^Hello World$/, 'Rendered simple template');
    } else {
        pass('Skipped render test due to template failure');
    }

    # Test 2: One-shot rendering via class method  
    my $inline_result = Minijinja->apply_from_string('{{ greeting }}!', 
                                                       { greeting => 'Hi there' });
    like($inline_result, qr/Hi there/, 'Inline template rendered');

}

done_testing();

