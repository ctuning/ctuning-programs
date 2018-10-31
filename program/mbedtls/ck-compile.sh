#!/bin/sh

NAME="mbedtls-mbedtls-2.4.1"

rm -rf "$NAME"
tar -xvzf "../$NAME.tar.gz" >/dev/null 2>&1
cp ../benchmark.c "$NAME/programs/test/benchmark.c"

#cp ../MainAr.cpp "$NAME/CPP/7zip/UI/Console/MainAr.cpp"

cd "$NAME"
chmod +x scripts/config.pl

export CC=${CK_CC}
export CXX=${CK_CXX}
export CFLAGS="${CK_PROG_COMPILER_VARS} -I${CK_ENV_LIB_RTL_XOPENME_INCLUDE} ${CK_PROG_COMPILER_FLAGS} -static "
export LDFLAGS="${CK_PROG_LINKER_LIBS} ${CK_EXTRA_LIB_M}"
export LIBS="${CK_PROG_LINKER_LIBS} ${CK_EXTRA_LIB_M}"

make

cp programs/test/benchmark ../a.out

ls -1 library/ | grep ^ici_features_function.* | while read LINE
do
  cp library/$LINE   ../
done


