#!/bin/bash
set -euo pipefail
for name in DEVELOPER_ID_P12_BASE64 DEVELOPER_ID_P12_PASSWORD NOTARY_KEY_BASE64 NOTARY_KEY_ID NOTARY_ISSUER_ID; do
  if [[ -z "${!name:-}" ]]; then echo "Missing required signing secret: $name" >&2; exit 1; fi
done
keychain="$RUNNER_TEMP/carter-signing.keychain-db"
keychain_password="$(openssl rand -hex 32)"
printf '%s' "$DEVELOPER_ID_P12_BASE64" | base64 --decode > "$RUNNER_TEMP/developer-id.p12"
printf '%s' "$NOTARY_KEY_BASE64" | base64 --decode > "$RUNNER_TEMP/notary.p8"
chmod 600 "$RUNNER_TEMP/developer-id.p12" "$RUNNER_TEMP/notary.p8"
security create-keychain -p "$keychain_password" "$keychain"
security set-keychain-settings -lut 21600 "$keychain"
security unlock-keychain -p "$keychain_password" "$keychain"
security import "$RUNNER_TEMP/developer-id.p12" -P "$DEVELOPER_ID_P12_PASSWORD" -A -t cert -f pkcs12 -k "$keychain"
security list-keychains -d user -s "$keychain" "$HOME/Library/Keychains/login.keychain-db"
security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "$keychain_password" "$keychain" >/dev/null
identity="$(security find-identity -v -p codesigning "$keychain" | sed -n 's/.*"\(Developer ID Application:.*\)"/\1/p' | head -1)"
if [[ -z "$identity" ]]; then echo 'Certificate is not a valid Developer ID Application identity' >&2; exit 1; fi
printf 'SIGNING_IDENTITY=%s\n' "$identity" >> "$GITHUB_ENV"
