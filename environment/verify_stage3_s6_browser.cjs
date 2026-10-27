/** Actual browser controls against the real GPU container stack through Nginx. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');
const root = path.resolve(__dirname, '..');
const config = JSON.parse(fs.readFileSync(path.join(root, '.local/s6-browser-ready.json'), 'utf8'));
const base = config.base_url;
const folder = path.join(root, 'runs/stage3/S6', 'browser-' + Date.now());
fs.mkdirSync(folder, { recursive: true });
const report = { stage: 'S6', status: 'running', started_at: new Date().toISOString(),
  scope: 'Browser file input, preview, queued YOLO button, Timer result/gallery, private access and logout. '
       + 'Real GPU models in Docker via Redis/prefork and Nginx; zero external model calls.',
  checks: [], external_model_calls: 0, failed_responses: [], warnings: [] };

async function main() {
  const browser = await chromium.launch({ headless: true, channel: 'msedge' });
  try {
    const contexts = [];
    const pages = [];
    for (const username of ['teacher01', 'student01']) {
      const context = await browser.newContext({ baseURL: base, viewport: { width: 1440, height: 1080 } });
      contexts.push(context);
      await context.route('**/*', route => {
        const url = new URL(route.request().url());
        return url.origin === new URL(base).origin || ['data:', 'blob:'].includes(url.protocol)
          ? route.continue() : route.abort();
      });
      const page = await context.newPage();
      pages.push(page);
      page.on('response', async response => {
        if (response.status() >= 400 && response.url().includes('/gradio_api/')) {
          const url = new URL(response.url());
          if (url.pathname.endsWith('/upload_progress') && url.searchParams.get('upload_id') === 'undefined') {
            report.warnings.push('Gradio polls upload_progress with undefined id after upload; 404, actual upload and preview continue.');
            return;
          }
          let body = '';
          try { body = (await response.text()).slice(0, 200); } catch { /* Page may close after logout. */ }
          report.failed_responses.push({ path: url.pathname.split('=')[0], status: response.status(), body });
        }
      });
      await page.goto('/login');
      await page.getByLabel('账号', { exact: true }).fill(username);
      await page.getByLabel('密码', { exact: true }).fill(config.accounts.find(account => account.username === username).password);
      await page.getByRole('button', { name: '登录', exact: true }).click();
      await page.getByText('当前账号：' + username, { exact: true }).waitFor({ timeout: 30000 });
    }
    const teacher = pages[0];
    const uploadPromise = teacher.waitForResponse(response =>
      new URL(response.url()).pathname === '/ui/gradio_api/upload' && response.request().method() === 'POST');
    await teacher.locator('#source-file input[type=file]').setInputFiles(path.join(root, config.image));
    const upload = await uploadPromise;
    assert.equal(upload.status(), 200);
    const filename = (await upload.json())[0];
    const localButton = teacher.getByRole('button', { name: '运行 YOLOv8 + SAM 2', exact: true });
    await localButton.waitFor();
    await teacher.waitForFunction(() => !Array.from(document.querySelectorAll('button')).find(
      b => b.textContent.includes('运行 YOLOv8 + SAM 2'))?.disabled, { timeout: 30000 });
    await teacher.locator('#source-preview img').waitFor({ timeout: 15000 });
    report.checks.push('File input upload updates visible preview and enables button');
    await localButton.click();
    await teacher.locator('#run-status').getByText(/任务 .*已保存/).waitFor({ timeout: 15000 });
    assert.equal(await localButton.isDisabled(), true);
    await teacher.locator('#run-status').getByText(/任务 .*已完成/).waitFor({ timeout: 180000 });
    await teacher.locator('#local-gallery img').first().waitFor();
    assert.equal(await localButton.isEnabled(), true);
    report.checks.push('Button returns queued status; Timer renders completed gallery and enables button');
    const download = teacher.locator('#result-download a').first();
    const href = await download.getAttribute('href');
    assert.ok(href.includes('/api/v1/runs/'));
    const run_id = href.split('/')[4];
    assert.equal((await contexts[0].request.get('/api/v1/runs/' + run_id)).status(), 200);
    assert.equal((await contexts[1].request.get('/api/v1/runs/' + run_id)).status(), 404);
    assert.equal((await contexts[0].request.get(href)).status(), 200);
    assert.equal((await contexts[1].request.get(href)).status(), 404);
    const fileUrl = '/ui/gradio_api/file=' + encodeURI(filename.replaceAll('\\', '/'));
    assert.equal((await contexts[1].request.get(fileUrl)).status(), 404);
    report.checks.push('Result, ZIP and upload remain private to the logged-in owner');
    await teacher.screenshot({ path: path.join(folder, 'completed.png'), fullPage: true });
    await teacher.getByRole('tab', { name: '历史记录', exact: true }).click();
    await teacher.getByRole('button', { name: '刷新历史', exact: true }).click();
    await teacher.getByText(run_id, { exact: true }).first().waitFor();
    await teacher.getByRole('button', { name: '退出登录', exact: true }).click();
    await teacher.waitForURL('**/login');
    assert.equal((await contexts[0].request.get('/api/v1/runs/' + run_id)).status(), 401);
    report.checks.push('History survives completed action; logout revokes result access');
    assert.equal(report.failed_responses.filter(r => !r.path.includes('/file')).length, 0,
                 JSON.stringify(report.failed_responses));
    report.status = 'passed';
    report.browser = browser.version();
    report.screenshot = path.relative(root, path.join(folder, 'completed.png')).replaceAll('\\', '/');
  } finally {
    await browser.close();
  }
}

main().catch(error => {
  report.status = 'failed';
  report.error_type = error.name;
  report.error = String(error.message).replaceAll(root, '<project>').slice(0, 1800);
  process.exitCode = 1;
}).finally(() => {
  report.finished_at = new Date().toISOString();
  const body = JSON.stringify(report, null, 2) + '\n';
  fs.writeFileSync(path.join(root, 'reports/stage3-S6-browser-checks.json'), body);
  fs.writeFileSync(path.join(folder, 'checks.json'), body);
  fs.writeFileSync(path.join(root, '.local/s6-browser-done.json'), JSON.stringify({ status: report.status }));
  console.log(body);
});
