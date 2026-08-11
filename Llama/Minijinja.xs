/* Minijinja XS Module - Direct bindings to libminijinja_cabi.so */
#define PERL_NO_GET_CONTEXT
#include <EXTERN.h>
#include <perl.h>
#include <XSUB.h>

#ifdef MINIJINJA_INCLUDE_DIR
#  include <minijinja.h>
#else
#  error "MINIJINJA_INCLUDE_DIR must be defined via Makefile.PL INC"  
#endif


MODULE = Llama::Minijinja    PACKAGE = Llama::Minijinja

IV      /* Returns env handle or NULL if failed */
mj_env_new()
        CODE:
                RETVAL = (IV)mj_env_new();
        OUTPUT: 
                RETVAL

void    
mj_env_free(IV env)   /* struct mj_env* cast to IV */
        CODE:
                if ((struct mj_env*)env != NULL) {
                        mj_env_free((struct mj_env*)env);
                }

bool    /* Add template by name/source pair, returns success flag */
mj_add_template(IV env, char *name, char *source)  
        PREINIT:
                struct mj_env* e;
        CODE:
            e = (struct mj_env*)env;
            RETVAL = (e && name && source) ? mj_env_add_template(e, name, source) : false;
        OUTPUT:
                RETVAL


char*   
mj_render_simple(IV env_handle, const char *tmpl_name_or_src, SV *context_hash_sv)
        PREINIT:
            struct mj_env *env;
            HV *ctx_hv;
            HE *entry; 
            SV *key_sv, *val_sv;
            struct mj_value ctx_val;  /* Root object for context */
            STRLEN key_len, val_len;
            
        PPCODE:
            env = (struct mj_env*)env_handle;
            
            if (!env || !tmpl_name_or_src) {
                    XSRETURN_EMPTY;
                    return;
            }

            /* Create empty map to hold context variables */  
            ctx_val = mj_value_new_map();

            /* If we have a hash reference, populate it */  
            if (SvROK(context_hash_sv)) {
                    hv_iterinit((HV*)SvRV(context_hash_sv));
                    
                    while ((entry = hv_iternext((HV*)SvRV(context_hash_sv)))) {
                            key_sv = hv_iterkeysv(entry);
                            val_sv = hv_iterval((HV*)SvRV(context_hash_sv), entry);
                            
                            char *key_str = SvPV(key_sv, key_len);
                            struct mj_value item_val;
                            
                            /* Convert Perl value type to minijinja value */  
                            if (SvIOK(val_sv)) {      /* Integer/number */
                                    item_val = mj_value_new_i64(SvIV(val_sv));
                            } else if (SvNOK(val_sv)) {  /* Float/double */
                                    item_val = mj_value_new_f64(SvNV(val_sv)); 
                            } else if (SvPOK(val_sv)) {   /* String */
                                    char *strval = SvPV(val_sv, val_len);
                                    item_val = mj_value_new_string(strval);
                            } else if (val_sv == &PL_sv_undef) {  /* undef/null */
                                    item_val = mj_value_new_none();
                            } else {    /* Other types - use none as fallback for v1 */
                                    item_val = mj_value_new_none();
                            }
                            
                            /* Add this key-value pair to context map - consumes the value! */
                            bool ok = mj_value_set_string_key(&ctx_val, key_str, item_val);
                    }
            }

            /* Render template with our constructed context value */  
            const char *src = tmpl_name_or_src; 
            char *result_ptr = mj_env_render_named_str(env, "inline", src, ctx_val);
            
            /* Clean up: decrement reference on root context object we created */  
            struct mj_value cleanup_vcopy = ctx_val;
            mj_value_decref(&cleanup_vcopy);

            ST(0) = sv_2mortal(newSVpv(result_ptr ? result_ptr : "", strlen(result_ptr)));
            
            /* Free the Rust-allocated C string AFTER Perl copies its contents! */  
            if (result_ptr) mj_str_free(result_ptr);

            XSRETURN(1);


bool    
mj_error_exists()     /* Check if last operation set an error state */
        CODE:
                RETVAL = mj_err_is_set();
        OUTPUT:
                RETVAL

char*   /* Returns allocated error message - MUST be freed with str_free() */  
mj_last_error_msg()  
        CODE:
                RETVAL = mj_err_get_detail();
        OUTPUT:
                RETVAL
        
void    /* Free a C string returned by minijinja-cabi functions */  
mj_cstring_free(char *str) 
        CODE:
                if(str && str != (char*)NULL) {
                        mj_str_free(str);
                }



ENDXS; wc -l Llama/Minijinja.xs 2>/dev/null || ls Llama/*.xs | head -3
