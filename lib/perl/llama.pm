package llama;

use strict; use warnings;

use nginx;
use JSON::XS;
use POSIX qw(strftime);
use File::Path qw(make_path);

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
    my ($r) = @_;
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

            do_magic_fixes($r, $req, $slot_id);

            $rb = $_json->encode($req);
            #print_error("[debug] new:  ",$rb);

            # log modified request body with 5-min rotation
            eval {
                my $log_dir = "/tmp/request-logs";
                make_path($log_dir) unless -d $log_dir;
                my $t = time;
                my $min_block = int((localtime($t))[1] / 5) * 5;
                my $log_file = sprintf("%s/requests_%s_%02d.log",
                    $log_dir,
                    strftime("%Y%m%d_%H", localtime($t)),
                    $min_block);
                open(my $fh, ">>", $log_file) or do {
                    print_error("[WARN] cannot open log file $log_file: $!");
                    return;
                };
                my $timestamp = localtime();
                my $model = $req->{model} // "unknown";
                print_error("[INFO] Logging request body for model=$model to $log_file");
                (my $safe_rb = $rb) =~ s/[\r\n]+//g;
                print $fh "[$timestamp] method=$method slot_id=" . ($slot_id // "none") . " model=$model $safe_rb\n";
                close $fh;

                # Update latest symlink to point to current 5-min file
                my $link = "$log_dir/latest";
                unlink $link if -e $link;
                symlink($log_file, $link) or print_error("[WARN] cannot create symlink $link: $!");
            };
            if($@){
                print_error("[WARN] failed to log request body: $@");
            }
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

sub do_magic_fixes {
    my ($r, $llm_req, $slot_id) = @_;

    # add id_slot, make it a number
    $llm_req->{id_slot} = 0+$slot_id;

    # fix for qwen
    $llm_req->{messages}[0]{role} = "system";

    return;
}

sub print_error {
    print STDERR @_,"\n";
}

1;
