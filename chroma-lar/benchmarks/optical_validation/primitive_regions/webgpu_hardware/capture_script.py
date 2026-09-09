from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import threading, json
from playwright.sync_api import sync_playwright
server=ThreadingHTTPServer(('127.0.0.1',0),partial(SimpleHTTPRequestHandler,directory='/tmp/trichroma-webgpu-main'))
threading.Thread(target=server.serve_forever,daemon=True).start()
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True, executable_path='/sdf/home/y/youngsam/.cache/ms-playwright/chromium_headless_shell-1223/chrome-headless-shell-linux64/chrome-headless-shell', args=['--no-sandbox','--use-angle=vulkan','--use-vulkan=native','--ignore-gpu-blocklist','--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan','--enable-gpu','--enable-unsafe-webgpu','--enable-logging=stderr','--disable-gpu-watchdog'])
    page=browser.new_page(viewport={'width':1100,'height':1000})
    page.on('console',lambda message: print('console',message.type,message.text))
    page.on('pageerror',lambda error: print('pageerror',error))
    page.goto(f'http://localhost:{server.server_port}/?manual=1')
    print('ADAPTER',page.evaluate('async()=>await window.trichromaReady'))
    results=[]
    for name in ['theia','pixelTPC','pixelPads']:
        manifest=page.evaluate('async name=>await window.trichroma.loadScene(name+".json")',name)
        debug=page.evaluate('async()=>await window.trichroma.render({width:64,height:40,rays:2560,debug:true,jitter:false})')
        page.evaluate('async()=>await window.trichroma.render({rays:2500000})')
        frames=[page.evaluate('async seed=>await window.trichroma.render({rays:2500000,seed})',seed) for seed in range(3)]
        page.locator('#canvas').screenshot(path='/tmp/webgpu-hardware-'+name+'.png')
        results.append(dict(manifest=manifest,debug=debug,frames=frames))
        print('MEASURED',name,[f['milliseconds'] for f in frames],flush=True)
    open('/tmp/webgpu-hardware-final-frames.json','w').write(json.dumps(results))
    browser.close()
server.shutdown()
