@echo off

rem cM generated script

rem target_os_uoa: windows-generic-64

rem Prepare code dependencies
set CM_CODE_DEP_MVS2008=a1df9556a1b41389
call cm_code_env_a1df9556a1b41389.bat
set CM_CODE_DEP_MVS2008=a1df9556a1b41389
set CM_CODE_DEP_MVS2010=214afe3467bba868
call cm_code_env_214afe3467bba868.bat
set CM_CODE_DEP_MVS2010=214afe3467bba868

rem package: Intel Composer XE 2011 64bit environment with MVS2008 and MVS2010

set CM_2c8e90e7e3475680_INSTALL=D:\Work1\CM\cm-repos\ctuning-setup\.cmr\code\2c8e90e7e3475680

call "d:\Program Files (x86)\Intel\Composer XE 2011 SP1\bin\compilervars.bat" intel64 vs2008shell

set CM_CXX=icl
set CM_OBJ_EXT=.obj
set CM_CC=icl /Qstd=c99
set CM_FC=ifort /fpp
set CM_F90=ifort /fpp
set CM_F95=ifort /fpp
set CM_FLAGS_DLL=/MT /DWin /LD /FIwindows.h
set CM_FLAGS_DLL_EXTRA=/link /dll
set CM_FLAGS_OUTPUT=/Fe
set CM_DLL_EXT=.dll
set CM_LIB_EXT=.lib
set CM_FLAGS_CREATE_OBJ=/c
set CM_LB=lib
set CM_AR=lib
set CM_LB_OUTPUT=/OUT:
set CM_EXE_EXT=.exe
set CM_FLAGS_STATIC_BIN=/MT
set CM_FLAGS_STATIC_LIB=/MT
set CM_FLAGS_DYNAMIC_BIN=/MD
set CM_LD_DYNAMIC_FLAGS=/link /NODEFAULTLIB:LIBCMT
set CM_LD_FLAGS_EXTRA=
set CM_FLAGS_CREATE_ASM=/FA
set CM_ASM_EXT=.asm

set CM_MAKE=nmake
set CM_OBJDUMP=objdump -d
