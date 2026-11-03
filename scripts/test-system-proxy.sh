#!/bin/bash
# Run only on the disposable GitHub macOS VM; temporary network services are restored.
set -euo pipefail
work="$(mktemp -d)"
python3 scripts/proxy-fixture.py "$work/connect.log" > "$work/proxy.log" 2>&1 &
fixture_pid=$!
networksetup -listallnetworkservices | tail -n +2 | sed '/^\*/d' > "$work/services"
cleanup() {
  while IFS= read -r service; do
    sudo networksetup -setsecurewebproxystate "$service" off
    sudo networksetup -setwebproxystate "$service" off
    sudo networksetup -setautoproxystate "$service" off
  done < "$work/services"
  kill "$fixture_pid" 2>/dev/null || true
}
trap cleanup EXIT
for name in HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy; do
  unset "$name"
  launchctl unsetenv "$name" || true
done
# The runner starts with no managed proxies. Refuse to modify a configured machine.
while IFS= read -r service; do
  if networksetup -getsecurewebproxy "$service" | grep -q 'Enabled: Yes'; then echo 'Expected a clean CI VM' >&2; exit 1; fi
done < "$work/services"
while IFS= read -r service; do
  sudo networksetup -setsecurewebproxy "$service" 127.0.0.1 18081
  sudo networksetup -setsecurewebproxystate "$service" on
done < "$work/services"
# Open via LaunchServices, with no proxy environment. Exit status is checked through
# a private completion file because open -W itself does not propagate the app's status.
report="$work/static.result"
open -W -n -a "$PWD/dist/Carter.app" --args --check-downloads-report "$report"
test "$(cat "$report")" = OK
grep -q 'CONNECT huggingface.co' "$work/connect.log"
cp "$work/connect.log" dist/performance/system-proxy.log
: > "$work/connect.log"
while IFS= read -r service; do
  sudo networksetup -setsecurewebproxystate "$service" off
  sudo networksetup -setautoproxyurl "$service" http://127.0.0.1:18082/proxy.pac
  sudo networksetup -setautoproxystate "$service" on
done < "$work/services"
report="$work/pac.result"
open -W -n -a "$PWD/dist/Carter.app" --args --check-downloads-report "$report"
test "$(cat "$report")" = OK
grep -q 'CONNECT huggingface.co' "$work/connect.log"
cp "$work/connect.log" dist/performance/pac-proxy.log
