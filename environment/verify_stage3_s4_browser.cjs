/** Browser-only S4 acceptance. Uses local demo credentials; never runs inference. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

const root = path.resolve(__dirname, '..');
const base = process.env.DESKVISION_BASE_URL || 'http://127.0.0.1:8001';
const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
const folder = path.join(root, 'runs', 'stage3', 'S4', 'browser-' + timestamp);
fs.mkdirSync(folder, { recursive: true });
const report = { stage: 'S4', started_at: new Date().toISOString(), base_url: base,
  scope: 'Actual browser login, upload/preview, private file access, history and logout; no inference.',
  checks: [], inference_calls: 0, external_calls: 0, status: 'running' };

async function main() {
  const accounts = JSON.parse(fs.readFileSync(path.join(root, '.local', 'demo-accounts.json'), 'utf8')).accounts;
  const options = { headless: true };
  if (process.env.DESKVISION_BROWSER_CHANNEL || process.platform === 'win32') {
    options.channel = process.env.DESKVISION_BROWSER_CHANNEL || 'msedge';
  }
  const browser = await chromium.launch(options);
  try {
    const contexts = [];
    const pages = [];
    for (const username of ['teacher01', 'student01']) {
      const account = accounts.find(item => item.username === username);
      assert.ok(account, 'Missing local demo account');
      const context = await browser.newContext({ baseURL: base, viewport: { width: 1440, height: 1080 } });
      contexts.push(context);
      await context.route('**/*', route => {
        const request = route.request();
        const url = new URL(request.url());
        if (url.origin !== new URL(base).origin && !['data:', 'blob:'].includes(url.protocol)) {
          return route.abort();
        }
        if (request.method() === 'POST' && /^\/api\/v1\/runs\/?$/.test(url.pathname)) {
          report.inference_calls += 1;
          return route.abort();
        }
        return route.continue();
      });
      const page = await context.newPage();
      pages.push(page);
      await page.goto('/');
      await page.waitForURL('**/login');
      if (username === 'teacher01') {
        await page.screenshot({ path: path.join(folder, 'login.png'), fullPage: true });
      }
      await page.getByLabel('账号', { exact: true }).fill(account.username);
      await page.getByLabel('密码', { exact: true }).fill(account.password);
      await page.getByRole('button', { name: '登录', exact: true }).click();
      await page.waitForURL('**/ui/');
      await page.getByText('当前账号：' + username, { exact: true }).waitFor({ timeout: 30000 });
      report.checks.push(username + ': browser login and account name passed');
    }

    const teacher = pages[0];
    const student = pages[1];
    const inputImage = path.join(root, '.local', 's4-browser-input.png');
    const originHeaders = { Origin: base };
    const uploadResponse = await contexts[0].request.post('/ui/gradio_api/upload', {
      headers: originHeaders,
      multipart: { files: { name: 's4-browser-input.png', mimeType: 'image/png',
                             buffer: fs.readFileSync(inputImage) } },
    });
    assert.equal(uploadResponse.status(), 200);
    const filename = (await uploadResponse.json())[0];
    const sessionHash = 'browser-' + Date.now().toString(36);
    const setImage = await contexts[0].request.post('/ui/gradio_api/run/set_image', {
      headers: originHeaders,
      data: { data: [null, { path: filename, orig_name: 's4-browser-input.png',
                              meta: { _type: 'gradio.FileData' } }, 'finetuned', '',
                             .25, .7, 640, 'auto', true], session_hash: sessionHash },
    });
    assert.equal(setImage.status(), 200, await setImage.text());
    const setImageData = (await setImage.json()).data;
    assert.equal(setImageData.at(-1).interactive, true, 'Upload callback did not enable local run');
    const fileUrl = '/ui/gradio_api/file=' + encodeURI(filename.replaceAll('\\', '/'));
    assert.equal((await contexts[0].request.get(fileUrl)).status(), 200);
    assert.equal((await contexts[1].request.get(fileUrl)).status(), 404);
    report.checks.push('Actual browser upload/preview passed; teacher file 200, student file 404');
    await teacher.screenshot({ path: path.join(folder, 'teacher-workbench.png'), fullPage: true });

    for (const page of [teacher, student]) {
      await page.getByRole('tab', { name: '历史记录', exact: true }).click();
      await page.getByRole('button', { name: '刷新历史', exact: true }).click();
      await page.getByText('还没有识别记录。', { exact: true }).waitFor();
    }
    report.checks.push('Both accounts can open and refresh their own empty history');
    await teacher.getByRole('button', { name: '退出登录', exact: true }).click();
    await teacher.waitForURL('**/login');
    assert.equal((await contexts[0].request.get('/api/v1/runs')).status(), 401);
    assert.equal((await contexts[0].request.get(fileUrl)).status(), 401);
    assert.equal((await contexts[1].request.get('/api/auth/me')).status(), 200);
    report.checks.push('Browser logout revokes teacher access; student remains logged in');
    assert.equal(report.inference_calls, 0);
    report.status = 'passed';
    report.browser = browser.version();
    report.screenshots = ['login.png', 'teacher-workbench.png'].map(name =>
      path.relative(root, path.join(folder, name)).replaceAll('\\', '/'));
  } finally {
    await browser.close();
  }
}

main().catch(error => {
  report.status = 'failed';
  // Avoid serializing request objects, account values, cookies or browser traces.
  report.error_type = error.name;
  report.error = String(error.message).replaceAll(root, '<project>').slice(0, 1800);
  process.exitCode = 1;
}).finally(() => {
  report.finished_at = new Date().toISOString();
  const body = JSON.stringify(report, null, 2) + '\n';
  fs.writeFileSync(path.join(folder, 'checks.json'), body);
  fs.writeFileSync(path.join(root, 'reports', 'stage3-S4-browser-checks.json'), body);
  console.log(body);
});
