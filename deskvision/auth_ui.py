"""Small, same-origin login page for the browser workbench."""


LOGIN_PAGE = '''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>登录 · DeskVision</title>
  <style>
    :root{font-family:"Microsoft YaHei","Segoe UI",sans-serif;color:#17212b;background:#f4f7fa}
    *{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;padding:20px}
    main{width:min(100%,390px);background:#fff;border:1px solid #dce3e8;border-radius:6px;padding:30px}
    header{border-bottom:1px solid #e5eaee;padding-bottom:20px;margin-bottom:22px}
    header p{margin:0 0 8px;color:#2563eb;font-size:13px;font-weight:600}
    h1{font-size:23px;line-height:1.3;margin:0 0 6px}header span{font-size:14px;color:#596773}
    label{display:block;font-size:14px;font-weight:600;margin:16px 0 7px}
    input{width:100%;height:42px;border:1px solid #aebac4;border-radius:4px;padding:0 11px;font:inherit}
    input:focus{outline:3px solid #bfdbfe;border-color:#2563eb}
    button{width:100%;height:42px;margin-top:22px;border:0;border-radius:4px;background:#2563eb;color:#fff;font:inherit;font-weight:600;cursor:pointer}
    button:hover{background:#1d4ed8}button:focus-visible{outline:3px solid #93c5fd;outline-offset:2px}
    button:disabled{opacity:.65;cursor:wait}#message{min-height:22px;margin:12px 0 0;color:#b42318;font-size:14px}
    @media(max-width:420px){main{padding:23px}h1{font-size:21px}}
  </style>
</head>
<body><main>
  <header><p>DESKVISION · 桌面物品识别</p><h1>登录工作台</h1><span>使用管理员分配的账号进入。</span></header>
  <form id="login-form">
    <label for="username">账号</label><input id="username" name="username" autocomplete="username" required minlength="3" maxlength="32">
    <label for="password">密码</label><input id="password" name="password" type="password" autocomplete="current-password" required maxlength="128">
    <button id="submit" type="submit">登录</button><p id="message" role="status" aria-live="polite"></p>
  </form>
</main>
<script>
const form=document.getElementById('login-form');
const button=document.getElementById('submit');
const message=document.getElementById('message');
form.addEventListener('submit',async event=>{
  event.preventDefault();button.disabled=true;message.textContent='正在验证…';
  try{
    const response=await fetch('/api/auth/login',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({username:form.username.value,password:form.password.value})});
    if(!response.ok)throw new Error('账号或密码错误，请核对后重试。');
    window.location.replace('/ui/');
  }catch(error){message.textContent=error.message;button.disabled=false;}
});
</script></body></html>'''
