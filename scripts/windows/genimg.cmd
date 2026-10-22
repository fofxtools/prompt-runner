@echo off
rem genimg for a Windows VM: asks the server on the host (see docs/windows-vm.md).
rem Set the URL variable before calling to use another address.
setlocal
set PYTHONUTF8=1
if not defined GENIMG_URL set GENIMG_URL=http://192.168.56.1:8190
"%~dp0..\..\.venv\Scripts\python.exe" -m prompt_runner.genimg %*
