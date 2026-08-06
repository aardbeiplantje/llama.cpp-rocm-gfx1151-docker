use strict; use warnings; use ExtUtils::MakeMaker;
WriteMakefile( NAME=>"Llama", VERSION_FROM=>"Llama.pm", OBJECT=>"\$(O_FILES)", CC=>"g++", LD=>"g++", OPTIMIZE=>"-O2 -std=c++17", LIBS=>["-L../llama.cpp/build/bin","-Wl,-rpath,../llama.cpp/build/bin"], INC=>"-I../llama.cpp/include -I../llama.cpp/ggml/include" );
