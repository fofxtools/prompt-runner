# Using prompt-runner from a Windows VM

A virtual machine has no GPU, so the models run on the Linux **host**. The VM runs `genimg` and `llm` as clients of the
servers on the host: the images, replies and logs are written on the VM.

```
Windows VM (no GPU)                      Linux host (GPU)
genimg, llm  ------------ HTTP ------->  sd-server  (port 8190)
writes images, replies and logs here     llama server (port 8191)
```

This guide uses VirtualBox with a host-only network: the host is `192.168.56.1` and the VM is `192.168.56.101`.
For other setups, replace the addresses.

## Before you start

- **Host:** prompt-runner is installed and works ([Quickstart](../README.md#quickstart)).
- **VM:** Windows 10 or 11, Python 3.10+ and Git.
- **Network:** the VM can reach the host's host-only address (`192.168.56.1`). Step 4 below checks it.

## 1. Set up the host

### Let the VM reach the servers

By default a server listens on this computer only. Set `listen_host` in `config/config.yaml`:

| Value | Who can connect | Use it when |
|---|---|---|
| `127.0.0.1` | This computer only (the default if the key is missing) | You do not use a VM |
| `192.168.56.1` | The VM, over the host-only adapter. `localhost` does **not** work | You want to expose the servers to the VM and nothing else |
| `0.0.0.0` | Every address of the host: `localhost`, the VM and your network | No one else is on your network. The shipped config uses this |

> **Warning:** the servers have no login. Anyone who can connect can use your GPU. Do not use `0.0.0.0` on a shared or untrusted network.

A running server keeps the address it was started with. After changing the key, restart both:

```bash
genimg --stop; genimg --start
llm --stop;    llm --start
```

`genimg --status` and `llm --status` show the address each server uses. For one run you can override the key with
`GENIMG_HOST` and `LLM_HOST`.

If the VM is refused, check the host's firewall. For example, with `ufw`:

```bash
sudo ufw allow in on vboxnet0 to any port 8190,8191 proto tcp
```

### Load the models you want

The VM cannot start or switch models. Load them on the host:

```bash
genimg --start --model flux-2-klein-9b-Q4_K_M
llm --start qwen3.5-4b
```

## 2. Set up the VM

1. Get prompt-runner: `git clone`, or copy the folder without the host's `storage/` and `.venv`.
2. Create a virtual environment and install:

   ```bat
   cd prompt-runner
   python -m venv .venv
   .venv\Scripts\pip install -e .
   ```

3. Put the wrappers on your `PATH`. The folder `scripts\windows` has `genimg.cmd` and `llm.cmd`. Add it to your user `PATH`, then open a new terminal.

   In PowerShell (use the path of your clone):

   ```powershell
   $folder = "C:\Users\you\Documents\prompt-runner\scripts\windows"
   [Environment]::SetEnvironmentVariable("Path", [Environment]::GetEnvironmentVariable("Path", "User") + ";" + $folder, "User")
   ```

   Or with the dialog: Start menu, "Edit environment variables for your account", select `Path`, Edit, New, paste the folder, OK.

4. Check the connection:

   ```bat
   genimg --status
   llm --status
   ```

   You should see the model each server has loaded, for example:
   `running flux-2-klein-9b-Q4_K_M.gguf at http://192.168.56.1:8190 (remote; config entry: flux-2-klein-9b-Q4_K_M)`.

### What the `.cmd` files do

- Set `PYTHONUTF8=1`. Without it Windows reads and prints text as cp1252, which garbles dashes, accents and emoji.
- Set `GENIMG_URL=http://192.168.56.1:8190` and `LLM_URL=http://192.168.56.1:8191`, unless you already set them. Set them first to use another address.
- Run `.venv\Scripts\python.exe -m prompt_runner.genimg` (or `.llm`). They expect the virtual environment at `.venv` in the repo.

## 3. Use it

The commands are the same as on the host. See the [README](../README.md#llm-chat-with-a-local-model) for all options.

```bat
genimg "a red fox in the snow" -o fox.png
llm qwen3.5-4b "Say hi"
llm qwen3.5-4b "Describe this image." --image fox.png
```

Where things go:

- `genimg`: the image where you say, and the log in `.genimg\` beside it.
- `llm`: the log in `storage\results\llm-cli\` of the repo, or in `.llm\` beside the file with `-o`.

For an AI agent: `llm` prints the model's thinking on stderr. Use `--reasoning off` for quick answers, or add `2>nul` to drop the thinking from captured output.

## How remote mode behaves

"Remote mode" is on whenever `GENIMG_URL` or `LLM_URL` is set. It also works on Linux: `GENIMG_URL=http://host:8190 genimg ...`.

- **No starting, stopping or locking.** `--start` and `--stop` are refused with a message: do them on the host.
- **The loaded model decides.** The server reports which model file it has loaded.
  - `llm MODEL` and `genimg --model NAME` refuse another model, and print the command that loads it on the host.
  - Without `--model`, `genimg` uses the config entry whose file is loaded.
- **`--status` and `--list` ask the server.** `--list` shows no sizes, because the files are on the host.
- **Logs are written on the VM.** A row's `tags` say `server: remote` and the URL, and `engine` is the server's build.

## Limits

- The VM cannot switch models. Switch on the host.
- If the host switches models while the VM is working, that request fails with an ordinary error.
- A log row's `server_args` are the configured ones, not checked against the host.
- The server reports only the model file it has loaded, never a config entry's name. Entries that share a file (a model and its Lightning or LoRA variant) differ only in request settings, so one server serves them all: `--status` and `--list` show every one of them as loaded, and `genimg` without `--model` picks `default_image_model` if it is one of them, else the first. The check compares the model file only: a server started with other flags for the same file (another text encoder, say) is not noticed.
- Pressing Ctrl-C in a `.cmd` shows Windows' "Terminate batch job (Y/N)?" prompt. Answer `y`.
- There is no login or HTTPS: see the warning above.

## Troubleshooting

Run the diagnostics script on the VM. It uses only the standard library, changes nothing and sends no secrets:

```bat
python scripts\windows\diagnostics.py --host 192.168.56.1 --repo . --chat
```

Lines that start with `!!` are the ones to look at.

| Symptom | Likely cause | Fix |
|---|---|---|
| `cannot reach the server ... refused` | The server listens on `127.0.0.1` only, or is not running | Set `listen_host`, restart the servers ([above](#let-the-vm-reach-the-servers)) |
| `cannot reach the server ... timed out` | The host firewall blocks the port | Allow the ports on the host-only adapter |
| `the server at ... has X loaded, but 'Y' needs Z` | The host has another model loaded | Run the printed command on the host |
| `GENIMG_URL must look like http://...` | A malformed URL | Use `http://HOST:PORT`, no path |
| Garbled dashes, accents or emoji | `PYTHONUTF8` is not set | Run through the `.cmd` files, or set `PYTHONUTF8=1` |
| `No module named 'yaml'` | prompt-runner is not installed | `.venv\Scripts\pip install -e .` |
