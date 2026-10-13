# Shadow Gen: free text-to-video setup guide

Build your own free text-to-video generator. It runs on Google Colab's free T4 GPU, uses the open LTX-Video model, and gets a public link through ngrok. A single HTML file talks to it.

Cost: $0 (free Google account, free ngrok account, free Colab T4).

Honest status: the setup below loads the model and starts the server. Video generation on the free T4 is tight on memory. See **Known issues** before you start.

---

## What you will end up with

- A Colab notebook that loads `Lightricks/LTX-Video-0.9.7-distilled` on a T4 GPU and starts a Gradio server.
- An ngrok static domain (`your-name.ngrok-free.dev`) that points at that server.
- `ltx-generator.html`, a local page where you type a prompt and get a video.

Files in this repo:

| File | What it is |
|---|---|
| `README.md` | This guide |
| `ltx-generator.html` | The front end. You edit one line to point it at your domain |

---

## Step 1: ngrok account and static domain

1. Go to https://ngrok.com and sign up (free).
2. Dashboard → **Your Authtoken** → copy the token. Keep it private. Do not paste it in chat, screenshots, or public repos.
3. Dashboard → **Domains** → **+ New Domain**. Free accounts get one static domain. It looks like `something-random.ngrok-free.dev`. Copy it.

You will use two values from this step:
- `NGROK_TOKEN` (the authtoken)
- `NGROK_DOMAIN` (the static domain, without `https://`)

---

## Step 2: Google Colab notebook

1. Go to https://colab.research.google.com → **New notebook**. Rename it `ltx-backend`.
2. **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save.**
3. Create the cells below, in order. Use **+ Code** for each. Put a token or domain in its real value, not in the guide text.

### Cell 1: install dependencies

```python
import subprocess, sys
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q', '-U',
    'git+https://github.com/huggingface/diffusers',
    'transformers', 'accelerate', 'sentencepiece', 'gradio', 'pyngrok'
], check=True)
# torchao conflicts with diffusers git build and is not needed here
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=False)
print('deps ready')
```

Takes 1-3 minutes. If you see `cannot import name 'FqnToConfig' from 'torchao.quantization'` later, the torchao uninstall did not stick. Re-run this cell, then **Runtime → Restart session**, then run again from cell 1.

### Cell 2: ngrok authtoken

Replace `PASTE_YOUR_NGROK_TOKEN` with your authtoken.

```python
NGROK_TOKEN = 'PASTE_YOUR_NGROK_TOKEN'

from pyngrok import ngrok, conf
conf.get_default().auth_token = NGROK_TOKEN
print('ngrok token set')
```

Colab stores your notebook in your Google Drive. Do not share the notebook link with anyone while this token is in it. Better: use **Runtime → Manage sessions** or paste the token only when you run, and delete it from the cell afterward.

### Cell 3: check the GPU

```python
import torch
assert torch.cuda.is_available(), 'No GPU. Runtime → Change runtime type → T4 GPU'
gpu = torch.cuda.get_device_properties(0)
print(f'{gpu.name}  {gpu.total_memory/1e9:.1f} GB VRAM')
```

Expected: `Tesla T4  15.6 GB VRAM`.

### Cell 4: load the model

```python
import torch
from diffusers import LTXConditionPipeline

pipe = LTXConditionPipeline.from_pretrained(
    'Lightricks/LTX-Video-0.9.7-distilled',
    torch_dtype=torch.bfloat16
)
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()
print('Model ready')
```

Why `enable_sequential_cpu_offload()` and not `pipe.to('cuda')`: the full pipeline does not fit in the T4's 15 GB. `pipe.to('cuda')` throws `OutOfMemoryError`. Sequential offload moves one layer at a time to the GPU. It is slower, but it fits.

First run downloads roughly 35-50 GB of model files. Expect 20-40 minutes once. Later runs reuse the cache on the same runtime, but a new runtime (after a disconnect) downloads again.

### Cell 5: start the server

Replace `NGROK_DOMAIN_HERE` with your static domain (e.g. `something-random.ngrok-free.dev`, no `https://`).

```python
import gradio as gr
from pyngrok import ngrok
from diffusers.utils import export_to_video

NGROK_DOMAIN = 'NGROK_DOMAIN_HERE'
WIDTH, HEIGHT = 512, 320   # 704x480 runs out of memory on a T4

try:
    from diffusers import LTXVideoCondition
except ImportError:
    from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

def generate(prompt, negative_prompt, duration, ref_image=None):
    fps = 24
    num_frames = int(duration) * fps + 1
    conditions = None
    if ref_image is not None:
        conditions = [LTXVideoCondition(video=[ref_image], frame_index=0)]
    frames = pipe(
        conditions=conditions,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=WIDTH, height=HEIGHT,
        num_frames=num_frames,
        num_inference_steps=7,
        decode_timestep=0.05,
        guidance_scale=1.0,
        decode_noise_scale=0.025,
        output_type='pil',
    ).frames[0]
    out = '/tmp/ltx_out.mp4'
    export_to_video(frames, out, fps=fps)
    return out

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label='Prompt', lines=3),
        gr.Textbox(label='Negative Prompt',
                   value='worst quality, blurry, jittery, distorted, cartoon, text, logos'),
        gr.Dropdown(choices=[3, 4], value=4, label='Duration (seconds)'),
        gr.Image(label='Reference Image (optional)', type='pil'),
    ],
    outputs=gr.Video(label='Generated Video'),
    flagging_mode='never',
)

ngrok.kill()
tunnel = ngrok.connect(7860, domain=NGROK_DOMAIN)
print('\n' + '=' * 55)
print(f'Live at: {tunnel.public_url}')
print('=' * 55 + '\n')

demo.queue().launch(server_port=7860, share=False)
```

Notes on this cell:
- `flagging_mode='never'` is the current gradio name. Older gradio uses `allow_flagging='never'` and throws `TypeError` on newer versions.
- `WIDTH, HEIGHT = 512, 320` is the size that fits. Larger sizes run out of memory on a free T4.

### Run it

**Runtime → Run all** (or Ctrl+F9). Cells 1-5 run top to bottom. Cell 5 keeps running after it prints `Live at:`. That is normal: it is the server. Do not stop it.

---

## Step 3: the HTML front end

1. Open `ltx-generator.html` in a text editor (VS Code, Notepad++).
2. Find this line (around line 432):
   ```js
   const backendUrl = 'https://lining-moody-nervy.ngrok-free.dev';
   ```
   Replace the domain with **your** static domain:
   ```js
   const backendUrl = 'https://your-static-domain.ngrok-free.dev';
   ```
3. Also search for `lining-moody-nervy` and replace it everywhere (there is one more, the status label).
4. Save. Double-click the file to open it in Chrome.
5. Type a prompt, pick a duration, click **Execute**. Progress steps show: Infiltrating → Target acquired → Executing mission → Rendering frames → Mission complete.

---

## Known issues (read before you start)

| Problem | What you see | Fix / workaround |
|---|---|---|
| T4 out of memory | `OutOfMemoryError: CUDA out of memory` in cell 4 or cell 5 | Use `enable_sequential_cpu_offload()` (cell 4) and the 512x320 size (cell 5). Do not raise size or length without checking memory |
| Browser warning page | Opening the ngrok domain shows "You are about to visit..." | Click **Visit Site** once in that browser. Browsers cannot skip it on the streaming calls the HTML uses; if generations fail after that, see the next row |
| Streaming progress fails | Page says "Cannot reach backend" or stalls at Target acquired | The HTML streams progress with `EventSource`, which cannot send the `ngrok-skip-browser-warning` header. Free ngrok shows its warning page to it. Fix idea: switch the HTML to plain polling (not done in this repo yet) |
| Runtime disconnects | Colab shows "Runtime disconnected" after idle or a closed tab | Keep the Colab tab open while generating. Reconnect, then Run all again. First run after a disconnect re-downloads the model |
| "Saving failed / file updated in another tab" | Colab shows a warning | Keep only one Colab tab open on the notebook |
| `Error` in the HTML page, no traceback | The Gradio page shows an error box | Look at the Colab cell 5 output. The traceback is printed there |
| `pipe.to('cuda')` OOM | Cell 4 fails | Replace with `pipe.enable_sequential_cpu_offload()` |

---

## Security checklist

- Your ngrok authtoken gives full access to your ngrok account. If it leaks, go to the ngrok dashboard → **Authtokens** → **Regenerate**.
- Do not commit a notebook with a token in it to any public repo. Remove the token before saving or sharing.
- The static domain itself is not secret, but it lets anyone reach your GPU session while the notebook runs. Stop the runtime (**Runtime → Disconnect and delete runtime**) when you are done.

---

## Troubleshooting checklist

1. Cell 3 says no GPU → Runtime → Change runtime type → T4 GPU, then restart session.
2. Cell 1 errors on torchao → run cell 1 again, then Runtime → Restart session.
3. Cell 5 `TypeError ... allow_flagging` → use `flagging_mode='never'`.
4. Cell 5 prints `Live at:` but the HTML says error → open the ngrok domain in the same browser and click Visit Site, then try again.
5. Generation takes many minutes → normal on a free T4 with sequential offload.

---

## Credits and models

- Model: [Lightricks/LTX-Video-0.9.7-distilled](https://huggingface.co/Lightricks/LTX-Video-0.9.7-distilled) (check its license before any commercial use)
- Libraries: diffusers, transformers, accelerate, gradio, pyngrok
