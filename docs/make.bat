@ECHO OFF
set SPHINXBUILD=sphinx-build
set SOURCEDIR=source
set BUILDDIR=_build

if "%1" == "clean" (
  rmdir /S /Q %BUILDDIR%
  goto end
)

%SPHINXBUILD% -b html %SOURCEDIR% %BUILDDIR%\html

:end
