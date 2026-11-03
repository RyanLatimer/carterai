#!/bin/bash
set -euo pipefail
xcrun notarytool submit dist/Carter-Apple-Silicon.zip --key "$RUNNER_TEMP/notary.p8" --key-id "$NOTARY_KEY_ID" --issuer "$NOTARY_ISSUER_ID" --wait --timeout 30m --output-format json > dist/notarization.json
python3 - <<'PY'
import json
with open('dist/notarization.json') as f: result=json.load(f)
if result.get('status') != 'Accepted':
    raise SystemExit('Notarization was not accepted; refusing to publish. Inspect notarization.json.')
PY
xcrun stapler staple dist/Carter.app
xcrun stapler validate dist/Carter.app
spctl --assess --type execute --verbose=2 dist/Carter.app
ditto -c -k --sequesterRsrc --keepParent dist/Carter.app dist/Carter-Apple-Silicon.zip
