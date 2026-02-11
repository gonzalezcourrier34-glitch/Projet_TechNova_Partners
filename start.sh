#!/usr/bin/env bash
set -e
exec supervisord -n -c /etc/supervisor/supervisord.conf
