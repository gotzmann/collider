package main

/*
#cgo linux    CFLAGS: -O3 -std=c17   -I.          -fPIC -pthread -march=native -mtune=native -DNDEBUG -D_XOPEN_SOURCE=600 -D_GNU_SOURCE      -DLOG_DISABLE_LOGS
#cgo darwin   CFLAGS: -O3 -std=c17   -I.          -fPIC -pthread -mcpu=native                -DNDEBUG -D_XOPEN_SOURCE=600 -D_DARWIN_C_SOURCE -DHAVE_BUGGY_APPLE_LINKER -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64
#cgo linux  CXXFLAGS: -O3 -std=c++17 -I. -Icommon -fPIC -pthread -march=native -mtune=native -DNDEBUG -D_XOPEN_SOURCE=600 -D_GNU_SOURCE      -DLOG_DISABLE_LOGS
#cgo darwin CXXFLAGS: -O3 -std=c++17 -I. -Icommon -fPIC -pthread -mcpu=native                -DNDEBUG -D_XOPEN_SOURCE=600 -D_DARWIN_C_SOURCE -DHAVE_BUGGY_APPLE_LINKER -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64
#cgo linux   LDFLAGS: cpp/llama.o cpp/bridge.o cpp/janus.o cpp/ggml.o cpp/ggml-backend.o cpp/ggml-alloc.o cpp/ggml-quants.o cpp/sgemm.o cpp/unicode-data.o  cpp/unicode.o -lstdc++ -lm -lpthread -ldl -lrt -fopenmp
#cgo darwin  LDFLAGS: cpp/llama.o cpp/bridge.o cpp/janus.o cpp/ggml.o cpp/ggml-backend.o cpp/ggml-alloc.o cpp/ggml-quants.o -lstdc++ -framework Accelerate -framework Foundation
*/
import "C"
import "github.com/gotzmann/booster/pkg/booster"

func main() {
	booster.Run()
}
