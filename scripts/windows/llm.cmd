@echo off
rem llm for a Windows VM: asks the server on the host (see docs/windows-vm.md).
rem Set the URL variable before calling to use another address.
setlocal
set PYTHONUTF8=1
if not defined LLM_URL set LLM_URL=http://192.168.56.1:8191
"%~dp0..\..\.venv\Scripts\python.exe" -m prompt_runner.llm %*
