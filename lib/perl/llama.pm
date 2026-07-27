package llama;

use strict; use warnings;

use nginx;
use JSON::XS;

our $_json;

sub fixup {
    my ($r) = @_;
    my $r_ok = eval {
        $r->has_request_body(\&handle_req);
        return OK;
    };
    if($@){
        chomp(my $err = $@);
        print_error("[ERROR] request: $err");
        $r->send_http_header;
        $r->status(500);
        return OK;
    }
    return $r_ok;
}

sub handle_req {
    my ($r, $slot_id) = @_;
    my $method = $r->request_method;
    my $slot_id = $r->header_in("X-LLamaCPP-Id-slot");
    my $rb = $r->request_body();
    my $rf = $r->request_body_file();
    if(!defined $rb and defined $rf){
        open(my $b_fh, "<", $rf) or do {
            print_error("[ERROR] cant open tmp body file $rf: $!");
            return HTTP_INTERNAL_SERVER_ERROR;
        };
        local $/;
        $rb = <$b_fh>;
        close $b_fh;
    }
    $rb //= "{}";
    print_error("[INFO] REQUEST $method, header X-LLamaCPP-Id-slot: ".( $slot_id//"<no id_slot>"));
    if(length($slot_id) and $slot_id =~ m/^\d+$/){
        eval {
            #print_error("[debug] body: $rb, ".($rf//"<no file>"));
            my $req = JSON::XS::decode_json($rb);
            $_json //= JSON::XS->new->utf8->allow_blessed->allow_unknown->allow_nonref->convert_blessed->canonical;
            #print_error("[debug] orig: ",$_json->encode($req));
            print_error("[INFO] REQUEST $method, header X-LLamaCPP-Id-slot: ".( $slot_id//"<no id_slot>").", model:$req->{model}");
            $req->{id_slot} = 0+$slot_id;

            # fix for qwen
            $req->{messages}[0]{role} = "system";

            $rb = $_json->encode($req);
            #print_error("[debug] new:  ",$rb);
        };
        if($@){
            chomp(my $err = $@);
            print_error("[error] $err");
        }
    }
    $r->send_http_header("application/json");
    $r->print($rb);
    return OK;
}

sub print_error {
    print STDERR @_,"\n";
}

1;
