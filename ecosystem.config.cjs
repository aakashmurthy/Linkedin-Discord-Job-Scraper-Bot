const path = require('node:path');

module.exports = {
  apps: [{
    name: 'linkedin-job-bot',
    cwd: __dirname,
    script: path.join(__dirname, 'bot.py'),
    interpreter: path.join(__dirname, '.venv/bin/python'),
    instances: 1,
    exec_mode: 'fork',
    autorestart: true,
    restart_delay: 10000,
    min_uptime: '30s',
    max_restarts: 10,
    kill_timeout: 10000,
    // The application already writes rotating logs to discord.log.
    out_file: '/dev/null',
    error_file: '/dev/null',
    env: { PYTHONUNBUFFERED: '1' },
  }],
};
